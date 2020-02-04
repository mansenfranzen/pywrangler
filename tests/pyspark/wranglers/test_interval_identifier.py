"""This module contains tests for pyspark interval identifier.
isort:skip_file
"""

import pandas as pd
import pytest
from pywrangler.util.testing import PlainFrame

pytestmark = pytest.mark.pyspark  # noqa: E402
pyspark = pytest.importorskip("pyspark")  # noqa: E402

from tests.test_data.interval_identifier import (
    CollectionGeneral,
    CollectionIdenticalStartEnd,
    CollectionMarkerSpecifics,
    CollectionNoOrderGroupBy,
    MultipleIntervalsSpanningGroupbyExtendedTriple,
    ResultTypeRawIids,
    ResultTypeValidIids
)

from pywrangler.pyspark.wranglers.interval_identifier import (
    VectorizedCumSum,
    VectorizedCumSumAdjusted
)

WRANGLER = (VectorizedCumSum, VectorizedCumSumAdjusted)
WRANGLER_IDS = [x.__name__ for x in WRANGLER]
WRANGLER_KWARGS = dict(argnames='wrangler',
                       argvalues=WRANGLER,
                       ids=WRANGLER_IDS)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionGeneral.pytest_parametrize_kwargs("marker_use")
@CollectionGeneral.pytest_parametrize_testcases
def test_base(testcase, wrangler, marker_use):
    """Tests against all available wranglers and test cases.

    Parameters
    ----------
    testcase: DataTestCase
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.
    marker_use: dict
        Defines the marker start/end use.

    """

    # instantiate test case
    testcase_instance = testcase("pyspark")

    # instantiate wrangler
    kwargs = testcase_instance.test_kwargs.copy()
    kwargs.update(marker_use)
    wrangler_instance = wrangler(**kwargs)

    # pass wrangler to test case
    testcase_instance.test(wrangler_instance.transform)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionIdenticalStartEnd.pytest_parametrize_testcases
def test_identical_start_end(testcase, wrangler):
    """Tests against all available wranglers and test cases.

    Parameters
    ----------
    testcase: DataTestCase
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.

    """

    # instantiate test case
    testcase_instance = testcase("pyspark")

    # instantiate wrangler
    wrangler_instance = wrangler(**testcase_instance.test_kwargs)

    # pass wrangler to test case
    testcase_instance.test(wrangler_instance.transform)


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
    testcase_instance = testcase("pyspark")

    # instantiate wrangler
    wrangler_instance = wrangler(**testcase_instance.test_kwargs)

    # pass wrangler to test case
    testcase_instance.test(wrangler_instance.transform)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
def test_repartition(wrangler):
    """Tests that repartition has no effect.

    Parameters
    ----------
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.

    """

    # instantiate test case
    testcase_instance = MultipleIntervalsSpanningGroupbyExtendedTriple()

    # instantiate wrangler
    wrangler_instance = wrangler(**testcase_instance.test_kwargs)

    # pass wrangler to test case
    testcase_instance.test.pyspark(wrangler_instance.transform, repartition=5)


@pytest.mark.parametrize(**WRANGLER_KWARGS)
def test_result_type_raw_iids(wrangler):
    """Test for correct raw iids constraints. Returned result only needs to
    distinguish intervals regardless of their validity. Interval ids do not
    need to be in specific order.

    Parameters
    ----------
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.

    """

    testcase_instance = ResultTypeRawIids("pandas")
    wrangler_instance = wrangler(result_type="raw",
                                 **testcase_instance.test_kwargs)

    df_input = testcase_instance.input.to_pyspark()
    df_output = testcase_instance.output.to_pandas()
    df_result = wrangler_instance.transform(df_input)
    df_result = (PlainFrame.from_pyspark(df_result)
                 .to_pandas()
                 .sort_values(testcase_instance.orderby_columns)
                 .reset_index(drop=True))

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

    testcase_instance = ResultTypeValidIids("pyspark")
    kwargs = testcase_instance.test_kwargs.copy()
    kwargs.update(marker_use)
    wrangler_instance = wrangler(result_type="valid", **kwargs)

    df_input = testcase_instance.input.to_pyspark()
    df_output = testcase_instance.output.to_pandas()
    df_result = wrangler_instance.transform(df_input)
    df_result = (PlainFrame.from_pyspark(df_result)
                 .to_pandas()
                 .sort_values(testcase_instance.orderby_columns)
                 .reset_index(drop=True))

    col = testcase_instance.target_column_name
    pd.testing.assert_series_equal(df_result[col].diff().ne(0),
                                   df_output[col].diff().ne(0))

    pd.testing.assert_series_equal(df_result[col].eq(0),
                                   df_output[col].eq(0))


@pytest.mark.parametrize(**WRANGLER_KWARGS)
@CollectionNoOrderGroupBy.pytest_parametrize_testcases
def test_no_order_groupby(testcase, wrangler):
    """Tests correct behaviour for missing groupby columns.

    Parameters
    ----------
    testcase: DataTestCase
        Generates test data for given test case.
    wrangler: pywrangler.wrangler_instance.interfaces.IntervalIdentifier
        Refers to the actual wrangler_instance begin tested. See `WRANGLER`.

    """

    # instantiate test case
    testcase_instance = testcase("pyspark")

    # instantiate wrangler
    kwargs = testcase_instance.test_kwargs.copy()
    kwargs.update({'groupby_columns': None})
    wrangler_instance = wrangler(**kwargs)

    # pass wrangler to test case
    testcase_instance.test(wrangler_instance.transform)
