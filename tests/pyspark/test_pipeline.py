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
from pyspark.ml.param.shared import Param
from pyspark.ml import Transformer


def test_create_getter_setter():
    """Test correct creation of getter and setter methods for given name.

    """

    result = pipeline._create_getter_setter("Dummy")

    assert "getDummy" in result
    assert "setDummy" in result
    assert all([callable(x) for x in result.values()])

    class MockUp:
        def __init__(self):
            self.Dummy = "Test"

        def _set(self, **kwargs):
            return kwargs

        def getOrDefault(self, value):
            return value

    mock = MockUp()
    assert result["getDummy"](mock) == "Test"
    assert result["setDummy"](mock, 1) == {"Dummy": 1}


def test_create_param_dict():
    """Test correct creation of `Param` values and setter/getter methods.

    """

    param_keys = {"Dummy": 1}.keys()
    result = pipeline._create_param_dict(param_keys)

    members_callable = ["setParams", "getParams", "setDummy", "getDummy"]

    assert all([x in result for x in members_callable])
    assert all([callable(result[x]) for x in members_callable])
    assert "Dummy" in result
    assert isinstance(result["Dummy"], Param)

    class ParamMock:
        def __init__(self, name):
            self.name = name

    class MockUp:
        def _set(self, **kwargs):
            return kwargs

        def extractParamMap(self):
            return {ParamMock("key"): "value"}

    mock = MockUp()
    assert result["setParams"](mock, a=2) == {"a": 2}
    assert result["getParams"](mock) == {"key": "value"}


def test_instantiate_transformer():
    """Test correct instantiation of `Transformer` subclass instance.

    """

    params = {"Dummy": 2}
    dicts = pipeline._create_param_dict(params.keys())
    dicts.update({"attribute": "value"})

    instance = pipeline._instantiate_transformer("Name", dicts, params)

    assert instance.__class__.__name__ == "Name"
    assert issubclass(instance.__class__, Transformer)
    assert instance.getDummy() == 2


def test_wrangler_to_spark_transformer():
    """Test correct pyspark wrangler to `Transformer` conversion.

    """

    class DummyWrangler(PySparkSingleNoFit):
        """Test Doc"""
        def __init__(self, a=5):
            self.a = a

        def transform(self, number):
            return number + self.a

    stage_wrangler = concretize_abstract_wrangler(DummyWrangler)()

    instance = pipeline.wrangler_to_spark_transformer(stage_wrangler)

    assert issubclass(instance.__class__, Transformer)
    assert instance.__class__.__name__ == "DummyWrangler"
    assert instance.__doc__ == "Test Doc"
    assert instance.transform(10) == 15

    assert instance.geta() == 5
    instance.seta(10)
    assert instance.geta() == 10
    assert instance.transform(10) == 20


def test_func_to_spark_transformer():
    """Test correct python function to `Transformer` conversion.

    """

    def dummy(number, a=5):
        """Test Doc"""
        return number + a

    instance = pipeline.func_to_spark_transformer(dummy)

    assert issubclass(instance.__class__, Transformer)
    assert instance.__class__.__name__ == "dummy"
    assert instance.__doc__ == "Test Doc"
    assert instance.transform(10) == 15

    assert instance.geta() == 5
    instance.seta(10)
    assert instance.geta() == 10
    assert instance.transform(10) == 20


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
