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
    via command line or the PYWRANGLER_TEST_ENV environment variable.

    """

    tox_env = os.environ.get("PYWRANGLER_TEST_ENV", "").lower()

    # if master version, all tests are run, no skipping required
    if "master" in tox_env:
        return

    for skip_item in ("pyspark", "dask"):
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
