"""This module contains benchmarking utility for pandas wranglers.

"""

from typing import Union

from dask.dataframe import DataFrame

from pywrangler.benchmark import TimeProfiler
from pywrangler.wranglers.dask.base import DaskWrangler


class DaskTimeProfiler(TimeProfiler):
    """Approximate time that a dask wrangler instance requires to execute the
    `fit_transform` step.

    Please note, input dataframes are cached before timing execution to ensure
    timing measurements only capture wrangler's `fit_transform`. This may cause
    problems if the size of input dataframes exceeds available memory.

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

    def __init__(self, wrangler: DaskWrangler,
                 repetitions: Union[None, int] = None):
        self._wrangler = wrangler

        def wrapper(*args, **kwargs):
            """Wrapper function to call `compute()` to enforce computation.

            """

            wrangler.fit_transform(*args, **kwargs).compute()

        super().__init__(wrapper, repetitions)

    def profile(self, *dfs: DataFrame, **kwargs):
        """Profiles timing given input dataframes `dfs` which are passed to
        `fit_transform`.

        Please note, input dataframes are cached before timing execution to
        ensure timing measurements only capture wrangler's `fit_transform`.
        This may cause problems if the size of input dataframes exceeds
        available memory.

        """

        # cache input dataframes
        dfs_cached = [df.persist() for df in dfs]

        super().profile(*dfs_cached, **kwargs)

        # clear caches
        for df in dfs_cached:
            del df

        del dfs_cached

        return self
