"""Test spark base wrangler.

"""

import pytest

try:
    from pywrangler.wranglers.spark.base import SparkWrangler
except ImportError:
    SparkWrangler = None


@pytest.mark.pyspark
def test_spark_base_wrangler_engine():
    wrangler = SparkWrangler()

    assert wrangler.computation_engine == "spark"
