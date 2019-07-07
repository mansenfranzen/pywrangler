"""This module contains tests for pyspark benchmarks.

isort:skip_file
"""

import time

import pytest

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from pywrangler.pyspark.base import PySparkSingleNoFit
from pywrangler.pyspark.benchmark import PySparkTimeProfiler, \
    PySparkBaseProfiler
from pywrangler.util.testing import concretize_abstract_wrangler

SLEEP = 0.0001


@pytest.fixture
def wrangler_sleeps():
    class DummyWrangler(PySparkSingleNoFit):
        def transform(self, df):
            time.sleep(SLEEP)
            return df

    return concretize_abstract_wrangler(DummyWrangler)


def test_spark_time_profiler_fastest(spark, wrangler_sleeps):
    """Basic test for pyspark time profiler ensuring fastest timing is slower
    than forced sleep.

    """

    df_input = spark.range(10).toDF("col")

    time_profiler = PySparkTimeProfiler(wrangler_sleeps(), 1).profile(df_input)

    assert time_profiler.best >= SLEEP


def test_spark_time_profiler_no_caching(spark, wrangler_sleeps):
    df_input = spark.range(10).toDF("col")

    PySparkTimeProfiler(wrangler_sleeps(), 1).profile(df_input)

    assert df_input.is_cached is False


def test_spark_time_profiler_caching(spark, wrangler_sleeps):
    """Cache is released after profiling."""
    df_input = spark.range(10).toDF("col")

    PySparkTimeProfiler(wrangler_sleeps(), 1, cache_input=True)\
        .profile(df_input)

    assert df_input.is_cached is False


def test_spark_base_profiler_cache_input(spark):
    df = spark.range(10).toDF("col")

    PySparkBaseProfiler._cache_input([df])
    assert df.is_cached is True

    PySparkBaseProfiler._clear_cached_input([df])
    assert df.is_cached is False
