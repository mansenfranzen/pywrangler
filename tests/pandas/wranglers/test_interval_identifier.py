import pytest

from tests.test_data.interval_identifier import (
    CollectionGeneral,
    CollectionIdenticalStartEnd,
    CollectionFirstStartFirstEnd,
    CollectionFirstStartLastEnd,
    CollectionLastStartFirstEnd,
    CollectionLastStartLastEnd,
    MARKER_USE,
    MARKER_USE_KWARGS
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


@pytest.mark.parametrize(**MARKER_USE_KWARGS)
@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionGeneral.pytest_parametrize
def test_base(testcase, wrangler, marker_use):
    """Tests against all available wranglers and test cases .
    Parameters
    ----------
    test_case: function
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    marker_use: dict
        Defines the marker start/end use.
    """

    # instantiate test case
    testcase_instance = testcase("pandas")

    # instantiate wrangler
    wrangler_instance = wrangler(**testcase_instance.test_kwargs, **marker_use)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionIdenticalStartEnd.pytest_parametrize
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


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionFirstStartFirstEnd.pytest_parametrize
def test_first_start_first_end(testcase, wrangler):
    """Tests against all available wranglers and test cases.

    Parameters
    ----------
    testcase: DataTestCase
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.

    """

    # instantiate test case
    testcase_instance = testcase("pandas")

    # instantiate wrangler
    marker_use = MARKER_USE["FirstStartFirstEnd"]
    wrangler_instance = wrangler(**testcase_instance.test_kwargs, **marker_use)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionFirstStartLastEnd.pytest_parametrize
def test_first_start_last_end(testcase, wrangler):
    """Tests against all available wranglers and test cases.

    Parameters
    ----------
    testcase: DataTestCase
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.

    """

    # instantiate test case
    testcase_instance = testcase("pandas")

    # instantiate wrangler
    marker_use = MARKER_USE["FirstStartLastEnd"]
    wrangler_instance = wrangler(**testcase_instance.test_kwargs, **marker_use)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionLastStartFirstEnd.pytest_parametrize
def test_last_start_first_end(testcase, wrangler):
    """Tests against all available wranglers and test cases.

    Parameters
    ----------
    testcase: DataTestCase
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.

    """

    # instantiate test case
    testcase_instance = testcase("pandas")

    # instantiate wrangler
    marker_use = MARKER_USE["LastStartFirstEnd"]
    wrangler_instance = wrangler(**testcase_instance.test_kwargs, **marker_use)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionLastStartLastEnd.pytest_parametrize
def test_last_start_last_end(testcase, wrangler):
    """Tests against all available wranglers and test cases.

    Parameters
    ----------
    testcase: DataTestCase
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.

    """

    # instantiate test case
    testcase_instance = testcase("pandas")

    # instantiate wrangler
    marker_use = MARKER_USE["LastStartLastEnd"]
    wrangler_instance = wrangler(**testcase_instance.test_kwargs, **marker_use)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)
