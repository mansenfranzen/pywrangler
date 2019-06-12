"""Test spark base wrangler.

isort:skip_file
"""

import pytest

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from pywrangler.wranglers.spark.base import SparkWrangler


def test_spark_base_wrangler_engine():
    wrangler = SparkWrangler()

    assert wrangler.computation_engine == "spark"
