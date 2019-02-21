"""pytest configuration

"""

import os

import pytest


def pytest_addoption(parser):
    """Parser additional command line options.

    `--pyspark` and `--dask` flags are intended for local test invocation using
    pytest. pyspark and dask tests for Travis CI get triggered via the
    TSWRANGLER_TEST_ENV environment variable.

    """

    parser.addoption(
        "--pyspark",
        action="store_true",
        default=False,
        help="Run pyspark tests"
    )

    parser.addoption(
        "--dask",
        action="store_true",
        default=False,
        help="Run dask tests"
    )


def pytest_collection_modifyitems(config, items):
    """By default, pyspark and dask tests are skipped if not otherwise declared
    via command line or the TSWRANGLER_TEST_ENV environment variable.

    """

    for skip_item in ("pyspark", "dask"):

        tox_env = os.environ.get("TSWRANGLER_TEST_ENV", "").lower()
        run_env = skip_item in tox_env
        run_cmd = config.getoption("--{}".format(skip_item))

        # do not skip tests
        if run_env or run_cmd:
            continue

        # mark tests to be skipped
        reason = "{} test not activated".format(skip_item)
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if skip_item in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def spark_session():
    """Provide session wide Spark Session to avoid expensive recreation for
    each test.

    If pyspark is not available, skip tests.

    """

    try:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()
    except ImportError:
        pytest.skip("Pyspark not available.")
