"""pytest configuration

"""

import multiprocessing

import pytest

import pandas as pd


def patch_spark_create_dataframe():
    """Overwrite pyspark's default `SparkSession.createDataFrame` method to
    cache all test data. This has proven to be faster because identical data
    does not need to be converted multiple times. Out of memory should not
    occur because test data is very small and if a memory limit is reached,
    the oldest not-used dataframes should be dropped automatically.

    """

    from pyspark.sql.session import SparkSession

    cache = {}

    def wrapper(func):
        def wrapped(self, data, *args, schema=None, **kwargs):
            # create hashable key
            if isinstance(data, pd.DataFrame):
                key = tuple(data.columns), data.values.tobytes()
            else:
                key = str(data), schema

            # check existent result and return cached dataframe
            if key in cache:
                return cache[key]
            else:
                result = func(self, data, *args, schema=schema, **kwargs)
                result.cache()
                cache[key] = result
                return result

        return wrapped

    SparkSession.createDataFrame = wrapper(SparkSession.createDataFrame)


@pytest.fixture(scope="session")
def spark(request):
    """Provide session wide Spark Session to avoid expensive recreation for
    each test.

    If pyspark is not available, skip tests.

    """

    try:
        patch_spark_create_dataframe()

        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        # use pyarrow if available for pandas to pyspark communication
        spark.conf.set("pyspark.sql.execution.arrow.enabled", "true")

        # for testing, reduce the number of partitions to the number of cores
        cpu_count = multiprocessing.cpu_count()
        spark.conf.set("spark.sql.shuffle.partitions", cpu_count)

        # print pyspark ui url
        print("\nPySpark UiWebUrl:", spark.sparkContext.uiWebUrl, "\n")

        request.addfinalizer(spark.stop)
        return spark

    except ImportError:
        pytest.skip("Pyspark not available.")
