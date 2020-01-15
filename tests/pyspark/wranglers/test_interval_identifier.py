"""This module contains tests for pyspark interval identifier.

isort:skip_file
"""

import pytest

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from tests.test_data.interval_identifier import (
    BaseTests,
    IdenticalStartEndTests
)

from pywrangler.pyspark.wranglers.interval_identifier import VectorizedCumSum


WRANGLER = (VectorizedCumSum,)
WRANGLER_IDS = [x.__name__ for x in WRANGLER]
WRANGLER_KWARGS = dict(argnames='wrangler',
                       argvalues=WRANGLER,
                       ids=WRANGLER_IDS)

REPARTITION_KWARGS = dict(argnames='repartition',
                          argvalues=(None, 3))


@pytest.mark.parametrize(**REPARTITION_KWARGS)
@pytest.mark.parametrize(**WRANGLER_KWARGS)
@BaseTests.pytest_parametrize
def test_base(testcase, wrangler, repartition):
    """Tests against all available wranglers and test cases .

    Parameters
    ----------
    test_case: function
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    repartition: None, int
        Define repartition for input dataframe.

    """

    # instantiate test case
    testcase_instance = testcase("pyspark")

    # instantiate wrangler
    wrangler_instance = wrangler(**testcase_instance.test_kwargs)

    # pass wrangler to test case
    kwargs = dict(repartition=repartition)
    testcase_instance.test(wrangler_instance.transform, **kwargs)


@pytest.mark.parametrize(**REPARTITION_KWARGS)
@pytest.mark.parametrize(**WRANGLER_KWARGS)
@IdenticalStartEndTests.pytest_parametrize
def test_identical_start_end(testcase, wrangler,repartition):
    """Tests against all available wranglers and test cases .

    Parameters
    ----------
    test_case: function
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    repartition: None, int
        Define repartition for input dataframe.

    """

    # instantiate test case
    testcase_instance = testcase("pyspark")

    # instantiate wrangler
    wrangler_instance = wrangler(**testcase_instance.test_kwargs)

    # pass wrangler to test case
    kwargs = dict(repartition=repartition)
    testcase_instance.test(wrangler_instance.transform, **kwargs)