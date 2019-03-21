"""This module contains benchmarking utility.

"""

import gc
import inspect
import sys
from typing import Iterable, List

import numpy as np

from pywrangler.exceptions import NotProfiledError
from pywrangler.util import sanitizer
from pywrangler.util._pprint import enumeration, header, sizeof
from pywrangler.util.helper import cached_property


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


class BaseProfile:
    """Base class defining interface and providing common helper methods.

    By convention, the profiled object should always the be the first argument
    (ignoring self) passed to `__init__`. All public, relevant profiling
    metrics have to be defined as properties. All private attributes (methods
    and variables) need to start with an underscore.

    """

    def profile(self, *args, **kwargs):
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


class MemoryProfile(BaseProfile):
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

        self._check_is_profiled(['_max_usages', '_baselines'])

        return self._max_usages

    @property
    def baselines(self) -> List[int]:
        """Returns the absolute, baseline memory usages for each iteration in
        bytes. The baseline memory usage is defined as the memory usage before
        function execution.

        """

        self._check_is_profiled(['_max_usages', '_baselines'])

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


class PandasMemoryProfiler(BaseProfile):
    """Approximate memory usage for wrangler execution via `fit_transform`
    for given input dataframes.

    Computes the ratio of maximum memory usage and input memory usage as an
    estimate of how many times more memory is required for wrangler execution
    in regard to the input memory usage.

    """

    def __init__(self, wrangler, repetitions=5, precision=2):
        self._wrangler = wrangler
        self._repetitions = repetitions
        self._precision = precision

        self._memory_profile = None
        self._usage_input = None
        self._usage_output = None

    def profile(self, *dfs, **kwargs):

        memory_profile = MemoryProfile(self._wrangler.fit_transform,
                                       self._repetitions)
        self._memory_profile = memory_profile.profile(*dfs, **kwargs)

        self._usage_input = self._memory_usage_dfs(*dfs)

        dfs_output = self._wrangler.fit_transform(*dfs)
        dfs_output = sanitizer.ensure_tuple(dfs_output)
        self._usage_output = self._memory_usage_dfs(*dfs_output)

        return self

    @property
    def usage_increases_mean(self):
        """Returns the mean of the absolute memory increases across all
        iterations.

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
        """Returns the ratio of maximum memory usage and input memory usage.
        A value of 0 means no memory consumption during execution. A value of 1
        means that the wrangler additionally requires the same amount of the
        input memory usage during the `transform` step. A value of 2 means that
        the wrangler requires twice the amount of the input dataframes memory
        usage.

        """

        return self.usage_increases_mean / self.usage_input

    def report(self):
        """Profile memory usage via `profile` and provide human readable
        report.

        """

        # string part for header
        wrangler_name = self._wrangler.__class__.__name__
        str_header = header("{} - memory usage".format(wrangler_name))

        # string part for input and output dfs
        dict_dfs = {"Input dfs": sizeof(self.usage_input, self._precision),
                    "Ouput dfs": sizeof(self.usage_output, self._precision)}

        str_dfs = enumeration(dict_dfs, align_width=15, bullet_char="")

        # string part for transform/fit and ratio
        str_inc = sizeof(self.usage_increases_mean)
        str_std = sizeof(self._memory_profile.increases_std,
                         self._precision, width=0)
        str_inc += " (Std: {})".format(str_std)
        str_ratio = "{:>7.2f}".format(self.usage_ratio)
        dict_inc = {"Fit/Transform": str_inc,
                    "Ratio": str_ratio}

        str_inc = enumeration(dict_inc, align_width=15, bullet_char="")

        # build complete string and print
        template = "{}\n{}\n\n{}"
        report_string = template.format(str_header, str_dfs, str_inc)

        print(report_string)

    @staticmethod
    def _memory_usage_dfs(*dfs) -> int:
        """Return the memory usage in Bytes for all dataframes `dfs`.

        """

        mem_usages = [df.memory_usage(deep=True, index=True).sum()
                      for df in dfs]

        return int(np.sum(mem_usages))
