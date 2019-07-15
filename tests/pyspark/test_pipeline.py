"""This module tests the customized pyspark pipeline.

isort:skip_file
"""

import pytest

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from pyspark.sql import functions as F
from pywrangler.util.testing import concretize_abstract_wrangler
from pywrangler.pyspark import pipeline
from pywrangler.pyspark.base import PySparkSingleNoFit


def test_full_pipeline(spark):
    """Create two stages from PySparkWrangler and native function and check
    against correct end result of pipeline.

    """

    df_input = spark.range(10).toDF("value")
    df_output = df_input.withColumn("add1", F.col("value") + 1)\
                        .withColumn("add2", F.col("value") + 2)

    class DummyWrangler(PySparkSingleNoFit):
        def __init__(self, a=5):
            self.a = a

        def transform(self, df):
            return df.withColumn("add1", F.col("value") + 1)

    stage_wrangler = concretize_abstract_wrangler(DummyWrangler)()

    def stage_func(df, a=2):
        return df.withColumn("add2", F.col("value") + 2)

    pipe = pipeline.Pipeline([stage_wrangler, stage_func])
    test_result = pipe.transform(df_input)

    assert df_output.toPandas().equals(test_result.toPandas())
