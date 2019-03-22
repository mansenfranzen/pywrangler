"""This module contains benchmarking utility.

"""

import gc
import inspect
import sys
from typing import Iterable, List

import numpy as np
import pandas as pd

from pywrangler.exceptions import NotProfiledError
from pywrangler.util import sanitizer
from pywrangler.util._pprint import enumeration, header, sizeof
from pywrangler.util.helper import cached_property
from pywrangler.wranglers.pandas.base import PandasWrangler


def allocate_memory(size: float) -> np.ndarray:
    """Occupies memory by creating numpy array with given size (MB).

    Numpy is used deliberately to specifically define the used memory via
    dtype.

    Parameters
    ----------
    size: float
        Size in MB to be occupied.

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
    """Base class defining interface and providing common helper methods for
    memory and time profiler.

    By convention, the profiled object should always the be the first argument
    (ignoring self) passed to `__init__`. All public profiling metrics have to
    be defined as properties. All private attributes need to start with an
    underscore.

    """

    def profile(self, *args, **kwargs):
        """Contains the actual profiling implementation and should always
        return self.

        """

        raise NotImplementedError

    def report(self):
        """Creates basic report consisting the name of the profiler class, the
        name of the profiled object, and all defined metrics/properties.

        """

        # get name of profiler
        profiler_name = self.__class__.__name__

        # get name of profiled object
        parameters = inspect.signature(self.__init__).parameters.keys()
        profiled_object = getattr(self, '_{}'.format(list(parameters)[0]))

        try:
            profiled_obj_name = profiled_object.__name__
        except AttributeError:
            profiled_obj_name = profiled_object.__class__.__name__

        # get relevant metrics
        ignore = ('profile', 'report', 'profile_report')
        metric_names = [x for x in dir(self)
                        if not x.startswith('_')
                        and x not in ignore]
        metric_values = {x: getattr(self, x) for x in metric_names}

        print(header('{}: {}'.format(profiler_name, profiled_obj_name)), '\n',
              enumeration(metric_values), sep='')

    def profile_report(self, *args, **kwargs):
        """Calls profile and report in sequence.

        """

        self.profile(*args, **kwargs).report()

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

        if any([getattr(self, x) is None for x in attributes]):
            msg = ("This {}'s instance is not profiled yet. Call 'profile' "
                   "with appropriate arguments before using this method."
                   .format(self.__class__.__name__))

            raise NotProfiledError(msg)

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


class MemoryProfiler(BaseProfiler):
    """Approximate the maximum increase in memory usage when calling a given
    function. The maximum increase is defined as the difference between the
    maximum memory usage during function execution and the baseline memory
    usage before function execution.

    In addition, compute the mean increase in baseline memory usage between
    repetitions which might indicate memory leakage.

    The current solution is based on `memory_profiler` and is inspired by the
    IPython `%memit` magic which additionally calls `gc.collect()` before
    executing the function to get more stable results.

    Parameters
    ----------
    func: callable
        Callable object to be memory profiled.
    repetitions: int, optional
        Number of repetitions.

    """

    def __init__(self, func, repetitions=5):
        self._func = func
        self._repetitions = repetitions

        self._max_usages = None
        self._baselines = None

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
        mem_args = (self._func, args, kwargs)

        while counter < self._repetitions:
            gc.collect()
            baseline = memory_usage()[0]
            max_usage = memory_usage(mem_args, max_usage=True)[0]

            baselines.append(self._mb_to_bytes(baseline))
            max_usages.append(self._mb_to_bytes(max_usage))
            counter += 1

        self._max_usages = max_usages
        self._baselines = baselines

        return self

    @property
    def max_usages(self) -> List[int]:
        """Returns the absolute, maximum memory usages for each iteration in
        bytes.

        """

        self._check_is_profiled(['_max_usages'])

        return self._max_usages

    @property
    def baselines(self) -> List[int]:
        """Returns the absolute, baseline memory usages for each iteration in
        bytes. The baseline memory usage is defined as the memory usage before
        function execution.

        """

        self._check_is_profiled(['_baselines'])

        return self._baselines

    @property
    def increases(self) -> List[int]:
        """Returns the absolute memory increase for each iteration in bytes.
        The memory increase is defined as the difference between the maximum
        memory usage during function execution and the baseline memory usage
        before function execution.

        """

        return np.subtract(self.max_usages, self.baselines).tolist()

    @property
    def increases_mean(self) -> float:
        """Returns the mean of the absolute memory increases across all
        iterations.

        """

        return float(np.mean(self.increases))

    @property
    def increases_std(self) -> float:
        """Returns the standard variation of the absolute memory increases
        across all iterations.

        """

        return float(np.std(self.increases))

    @property
    def baseline_change(self) -> float:
        """Returns the mean change in baseline memory usage across all
        all iterations. The baseline memory usage is defined as the memory
        usage before function execution.
        """

        changes = np.diff(self.baselines)
        return float(np.mean(changes))


class PandasMemoryProfiler(BaseProfiler):
    """Approximate memory usage for pandas wrangler instances.

    Memory consumption is profiled while calling `fit_transform` for given
    input dataframes.

    As a key metric, `usage_ratio` is computed. It refers to the amount of
    memory which is required to execute the `fit_transform` step. More
    concretely, it estimates how much more memory is used standardized by the
    input memory usage (memory usage increase during function execution divided
    by memory usage of input dataframes). In other words, if you have a 1GB
    input dataframe, and the `usage_ratio` is 5, `fit_transform` needs 5GB free
    memory available to succeed. A `usage_ratio` of 0.5 given a 2GB input
    dataframe would require 1GB free memory available for computation.

    Parameters
    ----------
    wrangler: pywrangler.wranglers.pandas.base.PandasWrangler
        The wrangler instance to be profiled.
    repetitions: int
        The number of measurements for memory profiling.

    Attributes
    ----------
    usage_increases_mean: float
        The mean of the absolute memory increases across all iterations in
        bytes.
    usage_input: int
        Memory usage of input dataframes in bytes.
    usage_output: int
        Memory usage of output dataframes in bytes.
    usage_ratio: float
        The amount of memory required for computation in units of input
        memory usage.

    """

    def __init__(self, wrangler: PandasWrangler, repetitions: int = 5):
        self._wrangler = wrangler
        self._repetitions = repetitions

        self._memory_profile = None
        self._usage_input = None
        self._usage_output = None

    def profile(self, *dfs: pd.DataFrame, **kwargs):
        """Profiles the actual memory usage given input dataframes `dfs`
        which are passed to `fit_transform`.



        """

        # usage input
        self._usage_input = self._memory_usage_dfs(*dfs)

        # usage output
        dfs_output = self._wrangler.fit_transform(*dfs)
        dfs_output = sanitizer.ensure_tuple(dfs_output)
        self._usage_output = self._memory_usage_dfs(*dfs_output)

        # usage during fit_transform
        memory_profile = MemoryProfiler(self._wrangler.fit_transform,
                                        self._repetitions)
        self._memory_profile = memory_profile.profile(*dfs, **kwargs)

        return self

    @property
    def usage_increases_mean(self) -> float:
        """Returns the mean of the absolute memory increases across all
        iterations in bytes.

        """

        self._check_is_profiled(['_memory_profile'])
        return self._memory_profile.increases_mean

    @property
    def usage_input(self) -> float:
        """Returns the memory usage of the input dataframes in bytes.

        """

        self._check_is_profiled(['_usage_input'])
        return self._usage_input

    @property
    def usage_output(self) -> float:
        """Returns the memory usage of the output dataframes in bytes.

        """

        self._check_is_profiled(['_usage_output'])
        return self._usage_output

    @cached_property
    def usage_ratio(self) -> float:
        """Refers to the amount of memory which is required to execute the
        `fit_transform` step. More concretely, it estimates how much more
        memory is used standardized by the input memory usage (memory usage
        increase during function execution divided by memory usage of input
        dataframes). In other words, if you have a 1GB input dataframe, and the
        `usage_ratio` is 5, `fit_transform` needs 5GB free memory available to
        succeed. A `usage_ratio` of 0.5 given a 2GB input dataframe would
        require 1GB free memory available for computation.

        """

        return self.usage_increases_mean / self.usage_input

    def report(self):
        """Profile memory usage via `profile` and provide human readable
        report including memory usage of input and output dataframes, memory
        usage during `fit_transform`, the usage ratio and shows if
        the wrangler may have side effects in regard to memory consumption via
        the change in baseline memory usage.

        Returns
        -------
        None. Prints report to stdout.

        """

        enum_kwargs = dict(align_width=15, bullet_char="")

        # string part for header
        wrangler_name = self._wrangler.__class__.__name__
        str_header = header("{} - memory usage".format(wrangler_name))

        # string part for input and output dfs
        dict_dfs = {"Input dfs": sizeof(self.usage_input),
                    "Ouput dfs": sizeof(self.usage_output)}

        str_dfs = enumeration(dict_dfs, **enum_kwargs)

        # string part for transform/fit and ratio
        str_inc = sizeof(self.usage_increases_mean)
        str_std = sizeof(self._memory_profile.increases_std, width=0)
        str_inc += " (Std: {})".format(str_std)
        str_ratio = "{:>7.2f}".format(self.usage_ratio)
        str_baseline_change = sizeof(self._memory_profile.baseline_change)
        dict_inc = {"Fit/Transform": str_inc,
                    "Ratio": str_ratio,
                    "Baseline change": str_baseline_change}

        str_inc = enumeration(dict_inc, **enum_kwargs)

        # build complete string and print
        template = "{}\n{}\n\n{}"
        report_string = template.format(str_header, str_dfs, str_inc)

        print(report_string)

    @staticmethod
    def _memory_usage_dfs(*dfs: pd.DataFrame) -> int:
        """Return the memory usage in Bytes for all dataframes `dfs`.

        Parameters
        ----------
        dfs: pd.DataFrame
            The pandas dataframes for which memory usage should be computed.

        Returns
        -------
        memory_usage: int
            The computed memory usage in bytes.

        """

        mem_usages = [df.memory_usage(deep=True, index=True).sum()
                      for df in dfs]

        return int(np.sum(mem_usages))
