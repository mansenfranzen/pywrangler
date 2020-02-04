"""This module contains tests for DataTestCase.

"""
import pytest
from pywrangler.util.testing.datatestcase import DataTestCase, TestCollection


@pytest.fixture
def datatestcase():
    class TestCase(DataTestCase):

        def input(self):
            return self.output["col1"]

        def output(self):
            return {"col1:i": [1, 2, 3],
                    "col2:i": [2, 3, 4]}

        def mutants(self):
            return {("col1", 0): 10}

    return TestCase


def test_engine_tester(datatestcase):
    def test_func(df):
        return df

    # assert missing engine specification
    with pytest.raises(ValueError):
        datatestcase().test(test_func)

    # assert invalid engine
    with pytest.raises(ValueError):
        datatestcase().test(test_func, engine="not_exists")

    with pytest.raises(ValueError):
        datatestcase("not_exists").test(test_func)


def test_engine_tester_pandas(datatestcase):
    # test correct standard behaviour
    def test_func(df):
        df = df.copy()
        df["col2"] = df["col1"] + 1
        return df

    datatestcase("pandas").test(test_func)
    datatestcase().test(test_func, engine="pandas")
    datatestcase().test.pandas(test_func)

    # check merge input column
    def test_func(df):
        return df["col1"].add(1).to_frame("col2")

    datatestcase("pandas").test(test_func, merge_input=True)

    # pass kwargs with merge input
    def test_func(df, add, mul=0):
        return df["col1"].add(add).mul(mul).to_frame("col2")

    datatestcase("pandas").test(test_func,
                                test_kwargs={"mul": 1, "add": 1},
                                merge_input=True)


def test_engine_tester_pyspark(datatestcase):
    from pyspark.sql import functions as F

    def test_func(df):
        return df.withColumn("col2", F.col("col1") + 1)

    # test correct standard behaviour
    datatestcase("pyspark").test(test_func)

    # check repartition
    datatestcase("pyspark").test(test_func, repartition=2)

    # pass kwargs with repartition
    def test_func(df, add, mul=0):
        return df.withColumn("col2", (F.col("col1") + add) * mul)

    datatestcase("pyspark").test(test_func,
                                 test_kwargs={"add": 1, "mul": 1},
                                 repartition=2)


def test_engine_tester_surviving_mutant():
    """Tests for a mutant that does not killed and hence should raise an error.
    In this example, the mutant equals the actual correct input.
    """

    class TestCase(DataTestCase):
        def input(self):
            return self.output["col1"]

        def output(self):
            return {"col1:i": [1, 2, 3],
                    "col2:i": [2, 3, 4]}

        def mutants(self):
            return {("col1", 0): 1}

    def test_func(df):
        df = df.copy()
        df["col2"] = df["col1"] + 1
        return df

    with pytest.raises(AssertionError):
        TestCase().test.pandas(test_func)


def test_test_collection(datatestcase):
    collection = TestCollection([datatestcase])

    # test init
    assert collection.testcases == [datatestcase]
    assert collection.names == ["TestCase"]

    # test with custom parameter name
    parametrize = pytest.mark.parametrize
    param = dict(argvalues=[datatestcase], ids=["TestCase"], argnames="a")
    assert collection.pytest_parametrize_testcases("a") == parametrize(**param)

    # test with default parameter name
    param["argnames"] = "testcase"

    def func():
        pass

    assert (collection.pytest_parametrize_testcases(func) ==
            parametrize(**param)(func))

    # test test_kwargs
    kwargs = {"conf1": {"param1": 1, "param2": 2}}
    param = dict(argvalues=[1, 2], ids=["param1", "param2"], argnames="conf1")
    collection = TestCollection([datatestcase], test_kwargs=kwargs)
    assert (collection.pytest_parametrize_kwargs("conf1") ==
            parametrize(**param))

    with pytest.raises(ValueError):
        collection.pytest_parametrize_kwargs("notexists")
