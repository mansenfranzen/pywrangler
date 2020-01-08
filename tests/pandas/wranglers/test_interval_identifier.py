import pytest

from tests.test_data.interval_identifier import (
    BaseTests,
    IdenticalStartEndTests
)

from pywrangler.pandas.wranglers.interval_identifier import (
    NaiveIterator,
    VectorizedCumSum
)

pytestmark = pytest.mark.pandas

WRANGLER = (NaiveIterator, VectorizedCumSum)
WRANGLER_IDS = [x.__name__ for x in WRANGLER]
WRANGLER_KWARGS = dict(argnames='wrangler',
                       argvalues=WRANGLER,
                       ids=WRANGLER_IDS)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@BaseTests.pytest_parametrize
def test_base(testcase, wrangler):
    """Tests against all available wranglers and test cases .
    Parameters
    ----------
    test_case: function
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    """

    # instantiate test case
    testcase_instance = testcase("pandas")

    # instantiate wrangler
    wrangler_instance = wrangler(**testcase_instance.test_kwargs)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@IdenticalStartEndTests.pytest_parametrize
def test_identical_start_end(testcase, wrangler):
    """Tests against all available wranglers and test cases .
    Parameters
    ----------
    test_case: function
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    """

    # instantiate test case
    testcase_instance = testcase("pandas")

    # instantiate wrangler
    wrangler_instance = wrangler(**testcase_instance.test_kwargs)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)