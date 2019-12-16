"""This module tests the customized pyspark pipeline.

isort:skip_file
"""

import pytest

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from pyspark.sql import functions as F
from pywrangler.util.testing import concretize_abstract_wrangler
from pywrangler.pyspark import pipeline
from pywrangler.pyspark.pipeline import StageTransformerConverter
from pywrangler.pyspark.base import PySparkSingleNoFit
from pyspark.ml.param.shared import Param
from pyspark.ml import Transformer


@pytest.fixture
def pipe():
    """Create example pipeline

    """

    def add_1(df, a=2):
        return df.withColumn("add1", F.col("value") + a)

    def add_2(df, b=4):
        return df.withColumn("add2", F.col("value") + b)

    return pipeline.Pipeline([add_1, add_2])


def test_create_getter_setter():
    """Test correct creation of getter and setter methods for given name.

    """

    result = StageTransformerConverter._create_getter_setter("Dummy")

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

    converter = StageTransformerConverter(lambda x: None)
    param_keys = {"Dummy": 1}.keys()
    result = converter._create_param_dict(param_keys)

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

    converter = StageTransformerConverter(lambda x: None)

    params = {"Dummy": 2}
    dicts = converter._create_param_dict(params.keys())
    dicts.update({"attribute": "value"})

    instance = converter._instantiate_transformer("Name", dicts, params)

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
    instance = StageTransformerConverter(stage_wrangler).convert()

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

    instance = StageTransformerConverter(dummy).convert()

    assert issubclass(instance.__class__, Transformer)
    assert instance.__class__.__name__ == "dummy"
    assert instance.__doc__ == "Test Doc"
    assert instance.transform(10) == 15

    assert instance.geta() == 5
    instance.seta(10)
    assert instance.geta() == 10
    assert instance.transform(10) == 20

    # test passing a transformer already
    assert instance is StageTransformerConverter(instance).convert()

    # test passing invalid type
    with pytest.raises(ValueError):
        StageTransformerConverter(["Wrong Type"]).convert()


def test_pipeline_locator(spark, pipe):
    """Test index and label access for stages and dataframe representation.

    """

    df_input = spark.range(10).toDF("value")
    df_output = df_input.withColumn("add1", F.col("value") + 2) \
        .withColumn("add2", F.col("value") + 4)

    # test non existant transformer
    with pytest.raises(ValueError):
        pipe(Transformer())

    # test missing transformation
    with pytest.raises(ValueError):
        pipe(0)

    test_result = pipe.transform(df_input)

    stage_add_1 = pipe.stages[0]
    transform_add_1 = pipe._transformer.transformations[0]

    assert stage_add_1 is pipe[0]
    assert stage_add_1 is pipe["add_1"]
    assert stage_add_1 is pipe[stage_add_1]

    # test incorrect type
    with pytest.raises(ValueError):
        pipe(tuple())

    # test out of bounds error
    with pytest.raises(IndexError):
        pipe(20)

    # test ambiguous identifier
    with pytest.raises(ValueError):
        pipe("add")

    # test non existant identifier
    with pytest.raises(ValueError):
        pipe("I do not exist")

    assert transform_add_1 is pipe(0)
    assert transform_add_1 is pipe("add_1")

    assert test_result is pipe(1)
    assert test_result is pipe("add_2")

    assert df_output.toPandas().equals(test_result.toPandas())


def test_pipeline_cacher(spark, pipe):
    """Test pipeline caching functionality.

    """

    df_input = spark.range(10).toDF("value")

    # test empty cache
    assert pipe.cache.enabled == []

    # test disable on empty cache
    with pytest.raises(ValueError):
        pipe.cache.disable("add_1")

    pipe.cache.enable("add_2")
    pipe.transform(df_input)

    assert pipe("add_1").is_cached is False
    assert pipe("add_2").is_cached is True
    assert pipe.cache.enabled == [pipe["add_2"]]

    pipe.cache.enable(["add_1"])
    assert pipe("add_1").is_cached is True

    pipe.cache.disable("add_1")
    assert pipe("add_1").is_cached is False
    assert pipe("add_2").is_cached is True

    pipe.cache.clear()
    assert pipe.cache.enabled == []
    assert pipe("add_1").is_cached is False
    assert pipe("add_2").is_cached is False


def test_pipeline_transformer(spark, pipe):
    """Test correct pipeline transformation.

    """

    df_input = spark.range(10).toDF("value")

    assert bool(pipe._transformer) is False
    pipe.transform(df_input)
    assert bool(pipe._transformer) is True
    assert pipe._transformer.input_df is df_input

    assert [x for x in pipe._transformer] == pipe._transformer.transformations


def test_pipeline_profiler(spark):
    """Test pipeline profiler.

    """

    df_input = spark.range(10).toDF("value")

    def add_order(df):
        return df.withColumn("order", F.col("value") + 5)

    def add_groupby(df):
        return df.withColumn("groupby", F.col("value") + 10)

    def sort(df):
        return df.orderBy("order")

    def groupby(df):
        return df.groupBy("groupby").agg(F.max("value"))

    pipe = pipeline.Pipeline(stages=[add_order, add_groupby, sort, groupby])

    # test missing df
    with pytest.raises(ValueError):
        pipe.profile()

    # test non pipeline df before transform
    df_profiles = pipe.profile(df_input)

    assert df_profiles.loc[0, "name"] == "Input dataframe"
    assert df_profiles.loc[0, "rows"] == 10
    assert df_profiles.loc[0, "idx"] == "None"
    assert df_profiles.loc[1, "name"] == "add_order"
    assert df_profiles.loc[4, "stage_count"] == 3
    assert df_profiles.loc[4, "cols"] == 2
    assert df_profiles.loc[4, "cached"] == False # noqa E712

    # test pipeline profile after transform
    pipe.transform(df_input)

    df_profiles = pipe.profile()

    assert df_profiles.loc[0, "name"] == "Input dataframe"
    assert df_profiles.loc[0, "rows"] == 10
    assert df_profiles.loc[0, "idx"] == "None"
    assert df_profiles.loc[1, "name"] == "add_order"
    assert df_profiles.loc[4, "stage_count"] == 3
    assert df_profiles.loc[4, "cols"] == 2
    assert df_profiles.loc[4, "cached"] == False # noqa E712

    # add caching and test
    pipe.cache.enable(2)
    pipe.transform(df_input)

    df_profiles = pipe.profile()

    assert df_profiles.loc[3, "cached"] == True # noqa E712
    assert df_profiles.loc[4, "stage_count"] == 4


def test_pipeline_describer(spark):
    """Test pipeline describer.

    """

    df_input = spark.range(10).toDF("value")

    def add_order(df):
        return df.withColumn("order", F.col("value") + 5)

    def add_groupby(df):
        return df.withColumn("groupby", F.col("value") + 10)

    def sort(df):
        return df.orderBy("order")

    def groupby(df):
        return df.groupBy("groupby").agg(F.max("value"))

    pipe = pipeline.Pipeline(stages=[add_order, add_groupby, sort, groupby])

    # test missing df
    with pytest.raises(ValueError):
        pipe.profile()

    # test non pipeline df before transform
    df_descriptions = pipe.describe(df_input)

    assert df_descriptions.loc[0, "name"] == "Input dataframe"
    assert df_descriptions.loc[0, "idx"] == "None"
    assert df_descriptions.loc[1, "uid"] == pipe[0].uid
    assert df_descriptions.loc[1, "name"] == "add_order"
    assert df_descriptions.loc[1, "stage_count"] == 1
    assert df_descriptions.loc[4, "cols"] == 2
    assert df_descriptions.loc[4, "cached"] == False # noqa E712


def test_full_pipeline(spark):
    """Create two stages from PySparkWrangler and native function and check
    against correct end result of pipeline.

    """

    df_input = spark.range(10).toDF("value")
    df_output = df_input.withColumn("add1", F.col("value") + 1) \
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
