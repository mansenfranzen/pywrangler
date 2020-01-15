"""pytest configuration

"""

import multiprocessing

import pytest


@pytest.fixture(scope="session")
def spark(request):
    """Provide session wide Spark Session to avoid expensive recreation for
    each test.

    If pyspark is not available, skip tests.

    """

    try:
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
