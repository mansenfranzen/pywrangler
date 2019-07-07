"""pytest configuration

"""

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

        request.addfinalizer(lambda: spark.stop())
        return spark

    except ImportError:
        pytest.skip("Pyspark not available.")
