"""This module contains tests for DataTestCase.

"""
from pywrangler.util.testing.datatestcase import DataTestCase


def test_engine_tester_pandas():
    class Dummy(DataTestCase):
        def data(self):
            return {"col1:i": [1, 2, 3],
                    "col2:i": [2, 3, 4]}

        def input(self):
            return self.data["col1"]

        def output(self):
            return self.data

    def test_func(df):
        df = df.copy()
        df["col2"] = df["col1"] + 1
        return df

    Dummy("pandas").test(test_func)

    # check merge input column
    def test_func(df):
        return df["col1"].add(1).to_frame("col2")

    Dummy("pandas").test(test_func, merge_input=True)

    # pass args/kwargs
    def test_func(df, add, mul=0):
        return df["col1"].add(add).mul(mul).to_frame("col2")

    Dummy("pandas").test(test_func,
                         args=(1,),
                         kwargs={"mul": 1},
                         merge_input=True)

def test_engine_tester_pyspark():
    from pyspark.sql import functions as F

    class Dummy(DataTestCase):
        def data(self):
            return {"col1:i": [1, 2, 3],
                    "col2:i": [2, 3, 4]}

        def input(self):
            return self.data["col1"]

        def output(self):
            return self.data

    def test_func(df):
        return df.withColumn("col2", F.col("col1")+1)

    Dummy("pyspark").test(test_func)
    Dummy("pyspark").test(test_func, repartition=2)

    # pass args/kwargs
    def test_func(df, add, mul=0):
        return df.withColumn("col2", (F.col("col1")+add)*mul)

    Dummy("pyspark").test(test_func,
                          args=(1,),
                          kwargs={"mul": 1})

