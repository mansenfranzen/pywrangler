"""This module contains benchmarking utility for pandas wranglers.

"""

import gc
import sys
import warnings
from typing import Callable, List, Union

import numpy as np
from dask.diagnostics import ResourceProfiler

from pywrangler.benchmark import MemoryProfiler, TimeProfiler
from pywrangler.dask.base import DaskWrangler


class DaskBaseProfiler:
    """Define common methods for dask profiler.

    """

    def _wrap_fit_transform(self) -> Callable:
        """Wrapper function to call `compute()` on wrangler's `fit_transform`
        to enforce computation on lazily evaluated dask graphs.

        Returns
        -------
        wrapped: callable
            Wrapped `fit_transform` method as a function.

        """

        def wrapped(*args, **kwargs):
            return self.wrangler.fit_transform(*args, **kwargs).compute()

        return wrapped

    @staticmethod
    def _cache_input(dfs) -> List:
        """Persist lazily evaluated dask input collections before profiling to
        capture only relevant `fit_transform`.

        Parameters
        ----------
        dfs: iterable
            Dask collections which can be persisted.

        Returns
        -------
        persisted: iterable
            List of computed dask collections.

        """

        return [df.persist() for df in dfs]

    @staticmethod
    def _clear_cached_input(dfs):
        """Remove original reference to previously persisted dask collections
        to enable garbage collection to free memory. Explicitly check reference
        count and give warning if persisted dask collections are referenced
        elsewhere which would prevent memory deallocation.

        Parameters
        ----------
        dfs: iterable
            Persisted dask collections which should be removed.

        """

        # ensure reference counts are updated
        gc.collect()

        # check ref counts
        for df in dfs:
            if sys.getrefcount(df) > 3:
                warnings.warn("Persisted dask collection is referenced "
                              "elsewhere and prevents garbage collection",
                              ResourceWarning)

        dfs.clear()


class DaskTimeProfiler(TimeProfiler, DaskBaseProfiler):
    """Approximate time that a dask wrangler instance requires to execute the
    `fit_transform` step.

    Parameters
    ----------
    wrangler: pywrangler.wranglers.base.BaseWrangler
         The wrangler instance to be profiled.
    repetitions: None, int, optional
        Number of repetitions. If `None`, `timeit.Timer.autorange` will
        determine a sensible default.
    cache_input: bool, optional
        Dask collections may be cached before timing execution to ensure
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

    def __init__(self, wrangler: DaskWrangler,
                 repetitions: Union[None, int] = None,
                 cache_input: bool = False):
        self.wrangler = wrangler
        self.cache_input = cache_input

        func = self._wrap_fit_transform()
        super().__init__(func, repetitions)

    def profile(self, *dfs, **kwargs):
        """Profiles timing given input dataframes `dfs` which are passed to
        `fit_transform`.

        """

        if self.cache_input:
            dfs = self._cache_input(dfs)

        super().profile(*dfs, **kwargs)

        if self.cache_input:
            self._clear_cached_input(dfs)

        return self


class DaskMemoryProfiler(MemoryProfiler, DaskBaseProfiler):
    """Approximate memory usage that a dask wrangler instance requires to
    execute the `fit_transform` step.

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
        Dask collections may be cached before timing execution to ensure
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

    Notes
    -----
    The implementation uses dask's own `ResourceProfiler`.

    """

    def __init__(self, wrangler: DaskWrangler,
                 repetitions: Union[None, int] = 5,
                 interval: float = 0.01,
                 cache_input: bool = False):
        self.wrangler = wrangler
        self.cache_input = cache_input

        func = self._wrap_fit_transform()
        super().__init__(func, repetitions, interval)

    def profile(self, *dfs, **kwargs):
        """Profiles timing given input dataframes `dfs` which are passed to
        `fit_transform`.

        """

        if self.cache_input:
            dfs = self._cache_input(dfs)

        counter = 0
        baselines = []
        max_usages = []

        while counter < self.repetitions:
            gc.collect()

            with ResourceProfiler(dt=self.interval) as rprof:
                self.func(*dfs, **kwargs)

            mem_usages = [x.mem for x in rprof.results]
            baselines.append(np.min(mem_usages))
            max_usages.append(np.max(mem_usages))

            counter += 1

        self._max_usages = max_usages
        self._baselines = baselines
        self._measurements = np.subtract(max_usages, baselines).tolist()

        if self.cache_input:
            self._clear_cached_input(dfs)

        return self
