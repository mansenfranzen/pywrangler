"""This module contains benchmarking utility for pandas wranglers.

"""

import warnings
from typing import Callable, Iterable, Union

from pyspark.sql import DataFrame

from pywrangler.benchmark import MemoryProfiler, TimeProfiler
from pywrangler.wranglers.spark.base import SparkWrangler


class SparkBaseProfiler:
    """Define common methods for spark profiler.

    """

    def _wrap_fit_transform(self) -> Callable:
        """Wrapper function to call `count()` on wrangler's `fit_transform`
        to enforce computation on lazily evaluated spark dataframes.

        Returns
        -------
        wrapped: callable
            Wrapped `fit_transform` method as a function.

        """

        def wrapped(*args, **kwargs):
            return self.wrangler.fit_transform(*args, **kwargs).count()

        return wrapped

    @staticmethod
    def _cache_input(dfs: Iterable[DataFrame]):
        """Persist lazily evaluated spark dataframes before profiling to
        capture only relevant `fit_transform`. Apply `count` to enforce
        computation to create cached representation.

        Parameters
        ----------
        dfs: iterable
            Spark dataframes to be persisted.

        Returns
        -------
        persisted: iterable
            List of computed dask collections.

        """

        for df in dfs:
            df.persist()
            df.count()

    @staticmethod
    def _clear_cached_input(dfs: Iterable[DataFrame]):
        """Unpersist previously persisted spark dataframes after profiling.

        Parameters
        ----------
        dfs: iterable
            Persisted spark dataframes.

        """

        for df in dfs:
            df.unpersist()

            if df.is_cached:
                warnings.warn("Spark dataframe could not be unpersisted.",
                              ResourceWarning)


class SparkTimeProfiler(TimeProfiler, SparkBaseProfiler):
    """Approximate time that a spark wrangler instance requires to execute the
    `fit_transform` step.

    Parameters
    ----------
    wrangler: pywrangler.wranglers.base.BaseWrangler
         The wrangler instance to be profiled.
    repetitions: None, int, optional
        Number of repetitions. If `None`, `timeit.Timer.autorange` will
        determine a sensible default.
    cache_input: bool, optional
        Spark dataframes may be cached before timing execution to ensure
        timing measurements only capture wrangler's `fit_transform`. By
        default, it is disabled.

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

    def __init__(self, wrangler: SparkWrangler,
                 repetitions: Union[None, int] = None,
                 cache_input: bool = False):
        self.wrangler = wrangler
        self.cache_input = cache_input

        func = self._wrap_fit_transform()
        super().__init__(func, repetitions)

    def profile(self, *dfs: DataFrame, **kwargs):
        """Profiles timing given input dataframes `dfs` which are passed to
        `fit_transform`.

        Please note, input dataframes are cached before timing execution to
        ensure timing measurements only capture wrangler's `fit_transform`.
        This may cause problems if the size of input dataframes exceeds
        available memory.

        """

        if self.cache_input:
            self._cache_input(dfs)

        super().profile(*dfs, **kwargs)

        if self.cache_input:
            self._clear_cached_input(dfs)

        return self


class SparkMemoryProfiler(MemoryProfiler, SparkBaseProfiler):
    """Approximate memory usage that a spark wrangler instance requires to
    execute the `fit_transform` step.

    #TODO: provide implementation for profile

    Parameters
    ----------
    func: callable
        Callable object to be memory profiled.
    repetitions: int, optional
        Number of repetitions.
    interval: float, optional
        Defines interval duration between consecutive memory usage
        measurements in seconds.
    cache_input: bool, optional
        Spark dataframes may be cached before timing execution to ensure
        timing measurements only capture wrangler's `fit_transform`. By
        default, it is disabled.

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


    """

    def __init__(self, wrangler: SparkWrangler,
                 repetitions: Union[None, int] = 5,
                 interval: float = 0.01,
                 cache_input: bool = False):
        self.wrangler = wrangler
        self.cache_input = cache_input

        func = self._wrap_fit_transform()
        super().__init__(func, repetitions, interval)

    def profile(self, *dfs: DataFrame, **kwargs):
        """Profiles timing given input dataframes `dfs` which are passed to
        `fit_transform`.

        """

        raise NotImplementedError
