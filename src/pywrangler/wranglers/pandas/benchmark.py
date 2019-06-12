"""This module contains benchmarking utility for pandas wranglers.

"""

from typing import Union

import numpy as np
import pandas as pd

from pywrangler.benchmark import MemoryProfiler, TimeProfiler
from pywrangler.util import sanitizer
from pywrangler.wranglers.pandas.base import PandasWrangler


class PandasTimeProfiler(TimeProfiler):
    """Approximate time that a pandas wrangler instance requires to execute the
    `fit_transform` step.

    Parameters
    ----------
    wrangler: pywrangler.wranglers.base.BaseWrangler
         The wrangler instance to be profiled.
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

    """

    def __init__(self, wrangler: PandasWrangler,
                 repetitions: Union[None, int] = None):
        self._wrangler = wrangler
        super().__init__(wrangler.fit_transform, repetitions)


class PandasMemoryProfiler(MemoryProfiler):
    """Approximate memory usage that a pandas wrangler instance requires to
    execute the `fit_transform` step.

    As a key metric, `ratio` is computed. It refers to the amount of
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
    input: int
        Memory usage of input dataframes in bytes.
    output: int
        Memory usage of output dataframes in bytes.
    ratio: float
        The amount of memory required for computation in units of input
        memory usage.

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

    def __init__(self, wrangler: PandasWrangler, repetitions: int = 5,
                 interval: float = 0.01):
        self._wrangler = wrangler

        super().__init__(wrangler.fit_transform, repetitions, interval)

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
        super().profile(*dfs, **kwargs)

        return self

    @property
    def input(self) -> float:
        """Returns the memory usage of the input dataframes in bytes.

        """

        self._check_is_profiled(['_usage_input'])
        return self._usage_input

    @property
    def output(self) -> float:
        """Returns the memory usage of the output dataframes in bytes.

        """

        self._check_is_profiled(['_usage_output'])
        return self._usage_output

    @property
    def ratio(self) -> float:
        """Refers to the amount of memory which is required to execute the
        `fit_transform` step. More concretely, it estimates how much more
        memory is used standardized by the input memory usage (memory usage
        increase during function execution divided by memory usage of input
        dataframes). In other words, if you have a 1GB input dataframe, and the
        `usage_ratio` is 5, `fit_transform` needs 5GB free memory available to
        succeed. A `usage_ratio` of 0.5 given a 2GB input dataframe would
        require 1GB free memory available for computation.

        """

        return self.median / self.input

    @staticmethod
    def _memory_usage_dfs(*dfs: pd.DataFrame) -> int:
        """Return memory usage in bytes for all given dataframes.

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
