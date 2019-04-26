"""This module contains tests for spark benchmarks.

isort:skip_file
"""

import time

import pytest

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from pywrangler.wranglers.spark.base import SparkSingleNoFit
from pywrangler.wranglers.spark.benchmark import SparkTimeProfiler

SLEEP = 0.0001


@pytest.fixture
def wrangler_sleeps():
    class DummyWrangler(SparkSingleNoFit):
        def transform(self, df):
            time.sleep(SLEEP)
            return df

    return DummyWrangler


def test_spark_time_profiler_fastest(spark, wrangler_sleeps):
    """Basic test for spark time profiler ensuring fastest timing is slower
    than forced sleep.

    """

    df_input = spark.range(10).toDF("col")

    time_profiler = SparkTimeProfiler(wrangler_sleeps(), 1).profile(df_input)

    assert time_profiler.best >= SLEEP


def test_spark_time_profiler_no_caching(spark, wrangler_sleeps):
    """Pyspark input dataframes are cached during time profiling. Ensure input
    dataframes are released from caching after profiling.

    """

    df_input = spark.range(10).toDF("col")

    SparkTimeProfiler(wrangler_sleeps(), 1).profile(df_input)

    assert df_input.is_cached is False
