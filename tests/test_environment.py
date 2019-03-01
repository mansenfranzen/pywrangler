"""Check for working test environment.

"""

import os
import subprocess

import pytest


@pytest.mark.pyspark
def test_java_environment():
    """Pyspark requires Java to be available. It uses Py4J to start and
    communicate with the JVM. Py4J looks for JAVA_HOME or falls back calling
    java directly. This test explicitly checks for the java prerequisites for
    pyspark to work correctly. If errors occur regarding the instantiation of
    a spark session, this test helps to rule out potential java related causes.

    """

    java_home = os.environ.get("JAVA_HOME")

    java_version = subprocess.run(["java", "-version"],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  universal_newlines=True)

    if (java_home is None) and (java_version.returncode != 0):
        raise EnvironmentError("Java setup broken.")


@pytest.mark.pyspark
def test_pyspark_import():
    """Fail if pyspark can't be imported. This test is mandatory because other
    spark tests will be skipped if the spark session fixture fails.

    """

    try:
        import pyspark
        print(pyspark.__version__)
    except (ImportError, ModuleNotFoundError):
        pytest.fail("pyspark can't be imported")


@pytest.mark.pyspark
def test_pyspark_pandas_interaction(spark):
    """Check simple interaction between pyspark and pandes.

    """

    import pandas as pd
    import numpy as np

    df_pandas = pd.DataFrame(np.random.rand(10, 2), columns=["a", "b"])
    df_spark = spark.createDataFrame(df_pandas)
    df_converted = df_spark.toPandas()

    print("JJava:", os.environ.get["JAVA_HOME", "JAVA_HOME_NOT_FOUND"])

    pd.testing.assert_frame_equal(df_pandas, df_converted)
