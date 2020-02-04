import pytest
import pandas as pd

from tests.test_data.interval_identifier import (
    CollectionGeneral,
    CollectionIdenticalStartEnd,
    CollectionMarkerSpecifics,
    ResultTypeRawIids,
    ResultTypeValidIids,
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


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionGeneral.pytest_parametrize_kwargs("marker_use")
@CollectionGeneral.pytest_parametrize_testcases
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
    kwargs = testcase_instance.test_kwargs.copy()
    kwargs.update(marker_use)
    wrangler_instance = wrangler(**kwargs)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionIdenticalStartEnd.pytest_parametrize_testcases
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
@CollectionMarkerSpecifics.pytest_parametrize_testcases
def test_marker_specifics(testcase, wrangler):
    """Tests specific `marker_start_use_first` and `marker_end_use_first`
    scenarios.

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
    wrangler_instance = wrangler(**testcase_instance.test_kwargs)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)


@CollectionGeneral.pytest_parametrize_kwargs("marker_use")
@pytest.mark.parametrize(**WRANGLER_KWARGS)
def test_result_type_raw_iids(wrangler, marker_use):
    """Test for correct raw iids constraints. Returned result only needs to
    distinguish intervals regardless of their validity. Interval ids do not
    need to be in specific order.

    Parameters
    ----------
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    marker_use: dict
        Contains `marker_start_use_first` and `marker_end_use_first` parameters
        as dict.

    """

    testcase_instance = ResultTypeRawIids("pandas")
    kwargs = testcase_instance.test_kwargs.copy()
    kwargs.update(marker_use)

    wrangler_instance = wrangler(result_type="raw", **kwargs)

    df_input = testcase_instance.input.to_pandas()
    df_output = testcase_instance.output.to_pandas()
    df_result = wrangler_instance.transform(df_input)

    col = testcase_instance.target_column_name
    pd.testing.assert_series_equal(df_result[col].diff().ne(0),
                                   df_output[col].diff().ne(0))


@CollectionGeneral.pytest_parametrize_kwargs("marker_use")
@pytest.mark.parametrize(**WRANGLER_KWARGS)
def test_result_type_valid_iids(wrangler, marker_use):
    """Test for correct valid iids constraints. Returned result needs to
    distinguish valid from invalid intervals. Invalid intervals need to be 0.

    Parameters
    ----------
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    marker_use: dict
        Contains `marker_start_use_first` and `marker_end_use_first` parameters
        as dict.

    """

    testcase_instance = ResultTypeValidIids("pandas")
    kwargs = testcase_instance.test_kwargs.copy()
    kwargs.update(marker_use)

    wrangler_instance = wrangler(result_type="valid", **kwargs)

    df_input = testcase_instance.input.to_pandas()
    df_output = testcase_instance.output.to_pandas()
    df_result = wrangler_instance.transform(df_input)

    col = testcase_instance.target_column_name
    pd.testing.assert_series_equal(df_result[col].diff().ne(0),
                                   df_output[col].diff().ne(0))

    pd.testing.assert_series_equal(df_result[col].eq(0),
                                   df_output[col].eq(0))


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionNoOrderGroupBy.pytest_parametrize_kwargs("missing_order_group_by")
@CollectionNoOrderGroupBy.pytest_parametrize_testcases
def test_no_order_groupby(testcase, missing_order_group_by, wrangler):
    """Tests correct behaviour for missing groupby columns.

    Parameters
    ----------
    testcase: DataTestCase
        Generates test data for given test case.
    missing_order_group_by: dict
        Defines `orderby_columns` and `groupby_columns`.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.

    """

    # instantiate test case
    testcase_instance = testcase("pandas")

    # instantiate wrangler
    wrangler_kwargs = testcase_instance.test_kwargs.copy()
    wrangler_kwargs.update(missing_order_group_by)
    wrangler_instance = wrangler(**wrangler_kwargs)

    # pass wrangler to test case
    kwargs = dict(merge_input=True,
                  force_dtypes={"marker": testcase_instance.marker_dtype})
    testcase_instance.test(wrangler_instance.transform, **kwargs)
