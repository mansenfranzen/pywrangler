import pytest
import pandas as pd

from tests.test_data.interval_identifier import (
    CollectionGeneral,
    CollectionIdenticalStartEnd,
    CollectionFirstStartFirstEnd,
    CollectionFirstStartLastEnd,
    CollectionLastStartFirstEnd,
    CollectionLastStartLastEnd,
    ResultTypeRawIids,
    ResultTypeValidIids,
    MARKER_USE,
    MARKER_USE_KWARGS,
    CollectionNoOrderGroupBy)

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

@pytest.mark.parametrize(**MARKER_USE_KWARGS)
@pytest.mark.parametrize(**WRANGLER_KWARGS)
def test_result_type_raw_iids(wrangler, marker_use):
    """Test for correct raw iids constraints. Returned result only needs to
    distinguish intervals regardless of their validity. Interval ids do not
    need to be in specific order.

    """

    testcase_instance = ResultTypeRawIids("pandas")
    wrangler_instance = wrangler(result_type="raw",
                                 **testcase_instance.test_kwargs,
                                 **marker_use)

    df_input = testcase_instance.input.to_pandas()
    df_output = testcase_instance.output.to_pandas()
    df_result = wrangler_instance.transform(df_input)

    col = testcase_instance.target_column_name
    pd.testing.assert_series_equal(df_result[col].diff().ne(0),
                                   df_output[col].diff().ne(0))

@pytest.mark.parametrize(**MARKER_USE_KWARGS)
@pytest.mark.parametrize(**WRANGLER_KWARGS)
def test_result_type_valid_iids(wrangler, marker_use):
    """Test for correct valid iids constraints. Returned result needs to
    distinguish valid from invalid intervals. Invalid intervals need to be 0.

    """

    testcase_instance = ResultTypeValidIids("pandas")
    wrangler_instance = wrangler(result_type="valid",
                                 **testcase_instance.test_kwargs,
                                 **marker_use)

    df_input = testcase_instance.input.to_pandas()
    df_output = testcase_instance.output.to_pandas()
    df_result = wrangler_instance.transform(df_input)

    col = testcase_instance.target_column_name
    pd.testing.assert_series_equal(df_result[col].diff().ne(0),
                                   df_output[col].diff().ne(0))

    pd.testing.assert_series_equal(df_result[col].eq(0),
                                   df_output[col].eq(0))


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@pytest.mark.parametrize(**MARKER_USE_KWARGS)
@CollectionNoOrderGroupBy.pytest_parametrize_kwargs("missing_order_group_by")
@CollectionNoOrderGroupBy.pytest_parametrize
def test_no_order_groupby(testcase, missing_order_group_by, marker_use,
                          wrangler):
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
    wrangler_kwargs = testcase_instance.test_kwargs.copy()
    wrangler_kwargs.update(marker_use)
    wrangler_kwargs.update(missing_order_group_by)
    wrangler_instance = wrangler(**wrangler_kwargs)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)
