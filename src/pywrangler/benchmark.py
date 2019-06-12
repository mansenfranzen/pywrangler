"""This module contains benchmarking utility.

"""

import gc
import sys
import timeit
from typing import Callable, Iterable, List, Union

import numpy as np

from pywrangler.exceptions import NotProfiledError
from pywrangler.util._pprint import (
    enumeration,
    header,
    pretty_file_size,
    pretty_time_duration
)
from pywrangler.util.helper import get_param_names


def allocate_memory(size: float) -> np.ndarray:
    """Helper function to approximately allocate memory by creating numpy array
    with given size in MiB.

    Numpy is used deliberately to define the used memory via dtype.

    Parameters
    ----------
    size: float
        Size in MiB to be occupied.

    Returns
    -------
    memory_holder: np.ndarray

    """

    if size <= 0:
        return None

    empty_size = sys.getsizeof(np.ones(0))

    size_in_bytes = np.ceil(size * (2 ** 20)).astype(np.int64) - empty_size
    memory_holder = np.ones(size_in_bytes, dtype=np.int8)

    return memory_holder


class BaseProfiler:
    """Base class defining the interface for all profilers.

    Subclasses have to implement `profile` (the actual profiling method) and
    `less_is_better` (defining the ranking of profiling measurements).

    The private attribute `_measurements` is assumed to be set by `profile`.

    Attributes
    ----------
    measurements: list
        The actual profiling measurements.
    best: float
        The best measurement.
    median: float
        The median of measurements.
    worst: float
        The worst measurement.
    std: float
        The standard deviation of measurements.
    runs: int
        The number of measurements.

    Methods
    -------
    profile
        Contains the actual profiling implementation.
    report
        Print simple report consisting of best, median, worst, standard
        deviation and the number of measurements.
    profile_report
        Calls profile and report in sequence.

    """

    @property
    def measurements(self) -> List[float]:
        """Return measurements of profiling.

        """

        self._check_is_profiled(["_measurements"])

        return self._measurements

    @property
    def best(self) -> float:
        """Returns the best measurement.

        """

        if self.less_is_better:
            return np.min(self.measurements)
        else:
            return np.max(self.measurements)

    @property
    def median(self) -> float:
        """Returns the median of measurements.

        """

        return np.median(self.measurements)

    @property
    def worst(self) -> float:
        """Returns the worst measurement.

        """

        if self.less_is_better:
            return np.max(self.measurements)
        else:
            return np.min(self.measurements)

    @property
    def std(self) -> float:
        """Returns the standard deviation of measurements.

        """

        return np.std(self.measurements)

    @property
    def runs(self) -> int:
        """Return number of measurements.

        """

        return len(self.measurements)

    @property
    def less_is_better(self) -> bool:
        """Defines ranking of measurements.

        """

        raise NotImplementedError

    def profile(self, *args, **kwargs):
        """Contains the actual profiling implementation and has to set
        `self._measurements`. Always returns self.

        """

        raise NotImplementedError

    def report(self):
        """Print simple report consisting of best, median, worst, standard
        deviation and the number of measurements.

        """

        tpl = "{best} {sign} {median} {sign} {worst} ± {std} ({runs} runs)"

        fmt = self._pretty_formatter
        values = {"best": fmt(self.best),
                  "median": fmt(self.median),
                  "worst": fmt(self.worst),
                  "std": fmt(self.std),
                  "runs": self.runs,
                  "sign": "<" if self.less_is_better else ">"}

        print(tpl.format(**values))

    def profile_report(self, *args, **kwargs):
        """Calls profile and report in sequence.

        """

        self.profile(*args, **kwargs).report()

    def _pretty_formatter(self, value: float) -> str:
        """String formatter for human readable output of given input `value`.
        Should be replaced with sensible formatters for file size or time
        duration.

        Parameters
        ----------
        value: float
            Numeric value to be formatted.

        Returns
        -------
        pretty_string: str
            Human readable representation of `value`.

        """

        return str(value)

    def _check_is_profiled(self, attributes: Iterable[str]) -> None:
        """Check if `profile` was already called by ensuring passed attributes
        are not `None`.

        Parameters
        ----------
        attributes:
            Attribute name(s) given as string or a list/tuple of strings

        Returns
        -------
        None

        Raises
        ------
        NotProfiledError

        Notes
        -----
        Inspired by sklearns `check_is_fitted`.

        """

        if any([getattr(self, x, None) is None for x in attributes]):
            msg = ("This {}'s instance is not profiled yet. Call 'profile' "
                   "with appropriate arguments before using this method."
                   .format(self.__class__.__name__))

            raise NotProfiledError(msg)

    def __repr__(self):
        """Print representation of profiler instance.

        """

        # get name of profiler
        profiler_name = self.__class__.__name__

        # get parameter names
        param_names = get_param_names(self.__class__.__init__, ["self"])
        param_dict = {x: getattr(self, x) for x in param_names}

        return header(profiler_name) + enumeration(param_dict)


class MemoryProfiler(BaseProfiler):
    """Approximate the increase in memory usage when calling a given function.
    Memory increase is defined as the difference between the maximum memory
    usage during function execution and the baseline memory usage before
    function execution.

    In addition, compute the mean increase in baseline memory usage between
    repetitions which might indicate memory leakage.

    Parameters
    ----------
    func: callable
        Callable object to be memory profiled.
    repetitions: int, optional
        Number of repetitions.
    interval: float, optional
        Defines interval duration between consecutive memory usage
        measurements in seconds.

    Attributes
    ----------
    measurements: list
        The actual profiling measurements in bytes.
    best: float
        The best measurement in bytes.
    median: float
        The median of measurements in bytes.
    worst: float
        The worst measurement in bytes.
    std: float
        The standard deviation of measurements in bytes.
    runs: int
        The number of measurements.
    baseline_change: float
        The median change in baseline memory usage across all runs in bytes.

    Methods
    -------
    profile
        Contains the actual profiling implementation.
    report
        Print simple report consisting of best, median, worst, standard
        deviation and the number of measurements.
    profile_report
        Calls profile and report in sequence.

    Notes
    -----
    The implementation is based on `memory_profiler` and is inspired by the
    IPython `%memit` magic which additionally calls `gc.collect()` before
    executing the function to get more stable results.

    """

    def __init__(self, func: Callable, repetitions: int = 5,
                 interval: float = 0.01):
        self.func = func
        self.repetitions = repetitions
        self.interval = interval

    def profile(self, *args, **kwargs):
        """Executes the actual memory profiling.

        Parameters
        ----------
        args: iterable, optional
            Optional positional arguments passed to `func`.
        kwargs: mapping, optional
            Optional keyword arguments passed to `func`.

        """

        from memory_profiler import memory_usage

        counter = 0
        baselines = []
        max_usages = []

        func_args = (self.func, args, kwargs)
        mem_args = dict(interval=self.interval,
                        multiprocess=True,
                        max_usage=True)

        while counter < self.repetitions:
            gc.collect()
            baseline = memory_usage(**mem_args)
            max_usage = memory_usage(func_args, **mem_args)

            baselines.append(self._mb_to_bytes(baseline))
            max_usages.append(self._mb_to_bytes(max_usage[0]))
            counter += 1

        self._max_usages = max_usages
        self._baselines = baselines
        self._measurements = np.subtract(max_usages, baselines).tolist()

        return self

    @property
    def less_is_better(self) -> bool:
        """Less memory consumption is better.

        """

        return True

    @property
    def max_usages(self) -> List[int]:
        """Returns the absolute, maximum memory usages for each run in
        bytes.

        """

        self._check_is_profiled(['_max_usages'])

        return self._max_usages

    @property
    def baselines(self) -> List[int]:
        """Returns the absolute, baseline memory usages for each run in
        bytes. The baseline memory usage is defined as the memory usage before
        function execution.

        """

        self._check_is_profiled(['_baselines'])

        return self._baselines

    @property
    def baseline_change(self) -> float:
        """Returns the median change in baseline memory usage across all
        run. The baseline memory usage is defined as the memory usage
        before function execution.
        """

        changes = np.diff(self.baselines)
        return float(np.median(changes))

    def _pretty_formatter(self, value: float) -> str:
        """String formatter for human readable output of given input `value`.

        Parameters
        ----------
        value: float
            Numeric value to be formatted.

        Returns
        -------
        pretty_string: str
            Human readable representation of `value`.

        """

        return pretty_file_size(value)

    @staticmethod
    def _mb_to_bytes(size_mib: float) -> int:
        """Helper method to convert MiB to Bytes.

        Parameters
        ----------
        size_mib: float
            Size in MiB

        Returns
        -------
        size_bytes: int
            Size in bytes.

        """

        return int(size_mib * (2 ** 20))


class TimeProfiler(BaseProfiler):
    """Approximate the time required to execute a function call.

    By default, the number of repetitions is estimated if not set explicitly.

    Parameters
    ----------
    func: callable
        Callable object to be memory profiled.
    repetitions: None, int, optional
        Number of repetitions. If `None`, `timeit.Timer.autorange` will
        determine a sensible default.

    Attributes
    ----------
    measurements: list
        The actual profiling measurements in seconds.
    best: float
        The best measurement in seconds.
    median: float
        The median of measurements in seconds.
    worst: float
        The worst measurement in seconds.
    std: float
        The standard deviation of measurements in seconds.
    runs: int
        The number of measurements.

    Methods
    -------
    profile
        Contains the actual profiling implementation.
    report
        Print simple report consisting of best, median, worst, standard
        deviation and the number of measurements.
    profile_report
        Calls profile and report in sequence.

    Notes
    -----
    The implementation is based on standard library's `timeit` module.

    """

    def __init__(self, func: Callable, repetitions: Union[None, int] = None):
        self.func = func
        self.repetitions = repetitions

    def profile(self, *args, **kwargs):
        """Executes the actual time profiling.

        Parameters
        ----------
        args: iterable, optional
            Optional positional arguments passed to `func`.
        kwargs: mapping, optional
            Optional keyword arguments passed to `func`.

        """

        def wrapper():
            """Helper function without arguments which is passed to `repeat`
            which only calls given function with provided args and kwargs.

            """

            self.func(*args, **kwargs)

        timer = timeit.Timer(stmt=wrapper)

        if self.repetitions is None:
            repeat, _ = timer.autorange(None)
        else:
            repeat = self.repetitions

        self._measurements = timer.repeat(number=1, repeat=repeat)

        return self

    @property
    def less_is_better(self) -> bool:
        """Less time required is better.

        """

        return True

    def _pretty_formatter(self, value: float) -> str:
        """String formatter for human readable output of given input `value`.

        Parameters
        ----------
        value: float
            Numeric value to be formatted.

        Returns
        -------
        pretty_string: str
            Human readable representation of `value`.

        """

        return pretty_time_duration(value)
