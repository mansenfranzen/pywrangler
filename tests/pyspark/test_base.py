"""Test pyspark base wrangler.

isort:skip_file
"""

import pytest

from pywrangler.util.testing.util import concretize_abstract_wrangler

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from pywrangler.pyspark.base import PySparkWrangler


def test_spark_base_wrangler_engine():
    wrangler = concretize_abstract_wrangler(PySparkWrangler)()

    assert wrangler.computation_engine == "pyspark"
