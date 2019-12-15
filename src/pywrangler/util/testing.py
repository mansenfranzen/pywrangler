"""This module contains testing utility.

"""
import collections
import copy
import numbers
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import tabulate
from numpy.testing import assert_equal
from pandas.api import types

try:
    import pyspark
except ImportError:
    pyspark = None


class NullValue:
    """Represents null values. Provides operator comparison functions to allow
    sorting which is required to determine row order of data tables.

    """

    def __str__(self):
        return "NULL"

    def __lt__(self, other):
        return self

    def __gt__(self, other):
        return other

    def __eq__(self, other):
        return isinstance(other, NullValue)

    def __ne__(self, other):
        return self.__eq__(other) is False


NaN = np.NaN
NULL = NullValue()

TYPE_ROW = List[Union[bool, int, float, str, datetime, NullValue]]
TYPE_DSTR = Dict[str, str]
TYPE_DTYPE_INPUT = Union[List[str], TYPE_DSTR]

PRIMITIVE_TYPES = {"bool": (bool, NullValue),
                   "int": (int, NullValue),
                   "float": (float, int, NullValue),
                   "str": (str, NullValue),
                   "datetime": (datetime, NullValue)}


class ConverterFromPySpark:
    """Convert pyspark dataframe into plain TestDataTable.

    """

    TYPE_MAPPING = {"smallint": "int",
                    "int": "int",
                    "bigint": "int",
                    "boolean": "bool",
                    "float": "float",
                    "double": "float",
                    "string": "str",
                    "timestamp": "datetime",
                    "date": "datetime"}

    def __init__(self, df: 'pyspark.sql.DataFrame'):
        self.df = df

    def __call__(self) -> 'TestDataTable':
        """Converts pyspark dataframe to TestDataTable. Several types are not
        supported including BinaryType, DecimalType, ByteType, ArrayType and
        MapType.

        Returns
        -------
        datatable: TestDataTable
            Converted dataframe.

        """

        data = list(map(self.convert_null, self.df.collect()))
        columns, dtypes = self.get_column_dtypes()

        return TestDataTable(data=data, columns=columns, dtypes=dtypes)

    def get_column_dtypes(self) -> Tuple[List[str]]:
        """Get column names and corresponding dtypes.

        """

        columns, pyspark_dtypes = zip(*self.df.dtypes)

        # check unsupported pyspark dtypes
        unsupported = set(pyspark_dtypes).difference(self.TYPE_MAPPING.keys())
        if unsupported:
            raise ValueError("Unsupported dtype encountered: {}. Supported"
                             "dtypes are: {}."
                             .format(unsupported, self.TYPE_MAPPING.keys()))

        dtypes = [self.TYPE_MAPPING[dtype] for dtype in pyspark_dtypes]

        return columns, dtypes

    @staticmethod
    def convert_null(values: Iterable) -> list:
        """Substitutes python `None` with NULL values.

        Parameters
        ----------
        values: iterable

        """

        return [x
                if x is not None
                else NULL
                for x in values]


class ConverterToPySpark:
    """Collection of pyspark conversion methods as a composite of
    TestDataColumn. It handles spark specifics like NULL as None and proper
    type matching.

    """

    def __init__(self, parent: 'TestDataColumn'):
        self.parent = parent

    @property
    def sanitized(self) -> list:
        """Replaces Null values with None to conform pyspark missing value
        convention.

        """

        return [None if x is NULL else x
                for x in self.parent.values]

    def __call__(self) -> Tuple['pyspark.sql.types.StructField', list]:
        """Main entry point for composite which returns appropriate
        `StructField` with corresponding values.

        """

        from pyspark.sql import types

        mapping = {"bool": types.BooleanType(),
                   "int": types.IntegerType(),
                   "float": types.DoubleType(),
                   "str": types.StringType(),
                   "datetime": types.TimestampType()}

        pyspark_type = mapping[self.parent.dtype]
        field = types.StructField(self.parent.name, pyspark_type)

        return field, self.sanitized


class ConverterFromPandas:
    """Convert pandas dataframe into plain TestDataTable.

    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self,
                 dtypes: Optional[TYPE_DTYPE_INPUT] = None) \
            -> 'TestDataTable':
        """Converts pandas dataframe to TestDataTable. Dtypes will be inferred
        from pandas dataframe. However, dtypes may be provided explicitly
        to overwrite inferred dtypes because pandas missing values (np.NaN)
        always casts to type float (e.g. bool or int with missings will be
        casted to float).

        Parameters
        ----------
        dtypes: list, dict, optional
            If list is provided, each value represents a dtype and maps to
            one column of the dataframe in given order. If dict is provided,
            keys refer to column names and values represent dtypes.

        Returns
        -------
        datatable: TestDataTable
            Converted dataframe.

        """

        dtypes_validated = self.get_forced_dtypes(dtypes)
        dtypes_validated.update(self.get_object_dtypes(dtypes_validated))
        dtypes_validated.update(self.get_inferred_dtypes(dtypes_validated))

        columns = self.df.columns.tolist()
        dtypes = [dtypes_validated[column] for column in columns]
        data = [self.convert_series(column, dtypes_validated[column])
                for column in columns]

        data = list(zip(*data))

        return TestDataTable(data=data,
                             columns=self.df.columns.tolist(),
                             dtypes=dtypes)

    def get_forced_dtypes(self, dtypes: TYPE_DTYPE_INPUT) -> TYPE_DSTR:
        """Validate user provided `dtypes` parameter.

        Parameters
        ----------
        dtypes: list, dict
            If list is provided, each value represents a dtype and maps to
            one column of the dataframe in order. If dict is provided, keys
            refer to column names and values represent dtypes.

        Returns
        -------
        dtypes_forced: dict
            Keys refer to column names and values represent dtypes.

        """

        if isinstance(dtypes, list):
            if len(dtypes) != self.df.shape[1]:
                raise ValueError("Length mismatch: Length of `dtypes` ({}) "
                                 "has to equal the number of columns ({})."
                                 .format(len(dtypes), self.df.shape[1]))

            dtypes_forced = dict(zip(self.df.columns, dtypes))

        elif isinstance(dtypes, dict):
            dtypes_forced = dtypes

        elif dtypes is not None:
            raise ValueError("Parameter `dtypes` has to be of type `list` or "
                             "`dict`. However, type `{}` is given."
                             .format(type(dtypes)))

        else:
            dtypes_forced = {}

        if dtypes_forced:
            for column, dtype in dtypes_forced.items():
                if column not in self.df.columns:
                    raise ValueError("Column `{}` does not exist. Available "
                                     "columns are: `{}`"
                                     .format(column, self.df.columns))

                if dtype not in PRIMITIVE_TYPES:
                    raise ValueError("Dtype `{}` is invalid. Valid dtypes "
                                     "are: {}."
                                     .format(dtype, PRIMITIVE_TYPES.keys()))

        return dtypes_forced

    def get_object_dtypes(self, dtypes_validated: TYPE_DSTR) -> TYPE_DSTR:
        """Inspect all columns of dtype object and ensure no mixed dtypes are
        present. Raises type error otherwise. Ignores columns for which dtypes
        are already explicitly set.

        Parameters
        ----------
        dtypes_validated: dict
            Represents already given column/dtype pairs. Keys refer to column
            names and values represent dtypes.

        Returns
        -------
        dtypes_object: dict
            Keys refer to column names and values represent dtypes.

        """

        dtypes_object = {}

        for column in self.df.columns:
            if column in dtypes_validated:
                continue

            if types.is_object_dtype(self.df[column]):
                dtypes_object[column] = self.inspect_dtype_object(column)

        return dtypes_object

    def get_inferred_dtypes(self, dtypes_validated: TYPE_DSTR) -> TYPE_DSTR:
        """Get all dtypes for columns which have not been provided, yet.
        Assumes that columns of dtype object are not present. Raises type error
        otherwise.

        Parameters
        ----------
        dtypes_validated: dict
            Represents already given column/dtype pairs. Keys refer to column
            names and values represent dtypes.

        Returns
        -------
        dtypes_inferred: dict
            Keys refer to column names and values represent dtypes.

        """

        dtypes_inferred = {}

        for column in self.df.columns:
            if column in dtypes_validated:
                continue

            dtypes_inferred[column] = self.inspect_dtype(self.df[column])

        return dtypes_inferred

    def convert_series(self, column: str, dtype: str) -> TYPE_ROW:
        """Converts a column of pandas dataframe into TestDataTable readable
        format with specified dtype (np.NaN to NULL, timestamps to
        datetime.datetime).

        Parameters
        ----------
        column: str
            Identifier for column.
        dtype: str
            Dtype identifier.

        Returns
        -------
        values: list
            Converted pandas series as plain python objects.

        """

        series = self.df[column]

        if dtype != "float":
            series = series.fillna(NULL)

        values = self.force_dtype(series, dtype)

        return values

    def inspect_dtype_object(self, column: str) -> str:
        """Inspect series of dtype object and ensure no mixed dtypes are
        present. Try to infer actual dtype after removing np.NaN distinguishing
        dtypes bool and str.

        Parameters
        ----------
        column: str
            Identifier for column.

        Returns
        -------
        dtype: str
            Inferred dtype as string.

        """

        series = self.df[column].dropna()

        # check for bool
        try:
            conv = pd.to_numeric(series)
            return self.inspect_dtype(conv)
        except ValueError:
            pass

        # check for mixed dtypes
        dtypes = {type(x) for x in series}
        if len(dtypes) > 1:
            raise TypeError("Column `{}` has mixed dtypes: {}. Currently, "
                            "this is not supported."
                            .format(column, dtypes))

        # check for string
        if isinstance(series[0], str):
            return "str"

        # raise if unsupported dtype is encountered
        raise TypeError("Column `{}` has dtype `{}` which is currently "
                        "not supported."
                        .format(column, type(series[0])))

    @staticmethod
    def inspect_dtype(series: pd.Series) -> str:
        """Get appropriate dtype of pandas series. Checks against bool, int,
        float and datetime. If dtype object is encountered, raises type error.

        Parameters
        ----------
        series: pd.Series
            pandas series column identifier.

        Returns
        -------
        dtype: str
            Inferred dtype as string.

        """

        mapping = {types.is_bool_dtype: "bool",
                   types.is_integer_dtype: "int",
                   types.is_float_dtype: "float",
                   types.is_datetime64_any_dtype: "datetime"}

        for check, result in mapping.items():
            if check(series):
                return result

        raise TypeError("Type is not understand for column '{}'. Allowed "
                        "types are bool, int, float, str and datetime."
                        .format(series.name))

    @staticmethod
    def force_dtype(series: pd.Series, dtype: str) -> TYPE_ROW:
        """Attempts to convert values to provided type.

        Parameters
        ----------
        series: pd.Series
            Values in pandas representation.
        dtype: str
            Dtype identifier.


        Returns
        -------
        values: list
            Converted pandas series as plain python objects.


        """

        conv_funcs = {"bool": bool,
                      "int": int,
                      "float": float,
                      "str": str,
                      "datetime": lambda x: pd.to_datetime(x).to_pydatetime()}

        conv_func = conv_funcs[dtype]

        return [conv_func(x) if not isinstance(x, NullValue) else NULL
                for x in series]


class ConverterToPandas:
    """Collection of pandas conversion methods as a composite of
    TestDataColumn. It handles pandas specifics likes the missing distinction
    between NULL and NaN.

    """

    def __init__(self, parent: 'TestDataColumn'):
        self.parent = parent
        self.requires_nan = parent.has_nan or parent.has_null

    @property
    def sanitized(self) -> list:
        """Replaces any Null values with np.NaN to conform pandas' missing
        value convention.

        """

        return [np.NaN if x is NULL else x
                for x in self.parent.values]

    def __call__(self) -> pd.Series:
        """Main entry point of composite which calls appropriate converter
        method corresponding to parent's dtype.

        """

        converter = {"datetime": self._convert_datetime,
                     "int": self._convert_int,
                     "bool": self._convert_bool}

        func = converter.get(self.parent.dtype, self._convert)

        return func()

    def _convert(self, dtype=None) -> pd.Series:
        """Generic converter for non special dtypes.

        """

        dtype = dtype or self.parent.dtype
        return pd.Series(self.sanitized, dtype=dtype, name=self.parent.name)

    def _convert_bool(self) -> pd.Series:
        """Handle dtype float upcast if missings are present.

        """

        if self.requires_nan:
            dtype = "float"
        else:
            dtype = "bool"

        return self._convert(dtype=dtype)

    def _convert_int(self) -> pd.Series:
        """Since pandas 0.24.0 exists `arrays.IntegerArray` which could be used
        as an nullable interger dtype. However, this is still experimental
        (0.25.3) and hence is not used yet.

        """

        if self.requires_nan:
            dtype = "float"
        else:
            dtype = "int"

        return self._convert(dtype=dtype)

    def _convert_datetime(self) -> pd.Series:
        """Pandas timestamp values have to be created via `pd.to_datetime` and
        can't be casted via `astype`.

        """

        series = pd.to_datetime(self.sanitized)
        series.name = self.parent.name

        return series


class TestDataColumn:
    """Represents a single column of a TestDataTable. Handles dtype specific
    preprocessing and performs type validation. In addition, it contains
    conversion methods for all supported computation engines.

    """

    def __init__(self, name: str, dtype: str, values: tuple):

        self.name = name
        self.dtype = dtype

        # preprocess
        if dtype == "float":
            values = self._preprocess_float(values)
        elif dtype == "datetime":
            values = self._preprocess_datetime(values)

        self.values = values

        # get null/nan flags
        self.has_null = any([x is NULL for x in values])
        self.has_nan = any([x is np.NaN for x in values])

        # sanity check for dtypes
        self._check_dtype()

        # add composite converters
        self.to_pandas = ConverterToPandas(self)
        self.to_pyspark = ConverterToPySpark(self)

    @staticmethod
    def _preprocess_datetime(values: tuple) \
            -> Tuple[Union[datetime, NullValue]]:
        """Convenience method to allow timestamps of various formats.

        """

        processed = [pd.Timestamp(x).to_pydatetime()
                     if not isinstance(x, NullValue)
                     else x
                     for x in values]

        return tuple(processed)

    @staticmethod
    def _preprocess_float(values: Tuple) -> Tuple[Union[float, NullValue]]:
        """Convenience method to ensure numeric values are casted to float.

        """

        processed = [float(x)
                     if isinstance(x, numbers.Number)
                     else x
                     for x in values]

        return tuple(processed)

    def _check_dtype(self):
        """Ensures correct type of all values. Raises TypeError.

        """

        allowed_types = PRIMITIVE_TYPES[self.dtype]

        for value in self.values:
            if not isinstance(value, allowed_types):
                raise TypeError("Value '{}' has invalid type '{}'. Allowed "
                                "types are: {}"
                                .format(value, type(value), allowed_types))


class EqualityAsserter:
    """Collection of equality assertions as a composite of TestDataTable. It
    contains equality tests in regard to number of rows, columns, dtypes etc.

    """

    def __init__(self, parent: 'TestDataTable'):
        self.parent = parent

    def __call__(self,
                 other: 'TestDataTable',
                 assert_column_order: bool = False,
                 assert_row_order: bool = False):
        """Main entry point for equality assertion. By default, no strict
        column nor row order is assumed but may be enabled.

        Parameters
        ----------
        other: TestDataTable
            Instance to assert equality against.
        assert_column_order: bool, optional
            If enabled, column order will be tested. Otherwise, column order
            does not matter for equality assertion.
        assert_row_order: bool, optional
            If enabled, row order will be tested. Otherwise, row order does not
            matter for equality assertion.

        """

        self._assert_shape(other)
        self._assert_column_names(other, assert_column_order)
        self._assert_dtypes(other)

        if not assert_row_order:
            order_left = self._get_row_order(self.parent)
            order_right = self._get_row_order(other)

        for column in self.parent.columns:
            left = self.parent._columns[column].values
            right = other._columns[column].values

            if not assert_row_order:
                left = [left[idx] for idx in order_left]
                right = [right[idx] for idx in order_right]

            msg = "column=" + column
            assert_equal(left, right, err_msg=msg)

    def _assert_shape(self, other: 'TestDataTable'):
        """Check for identical shape

        """

        if self.parent.n_rows != other.n_rows:
            raise AssertionError("Unequal number of rows: "
                                 "left {} vs. right {}"
                                 .format(self.parent.n_rows, other.n_rows))

        if self.parent.n_cols != other.n_cols:
            raise AssertionError("Unequal number of columns: "
                                 "left {} vs right {}"
                                 .format(self.parent.n_cols, other.n_cols))

    def _assert_column_names(self,
                             other: 'TestDataTable',
                             assert_column_order: bool):
        """Check for matching column names. Take column order into account if
        required.

        """

        if assert_column_order:
            enum = enumerate(zip(self.parent.columns, other.columns))
            for idx, (left, right) in enum:
                if left != right:
                    raise AssertionError(
                        "Mismatching column names at index {}: "
                        "left '{}' vs. right '{}'"
                        .format(idx + 1, left, right)
                    )
        else:
            left = set(self.parent.columns)
            right = set(other.columns)

            if left != right:
                left_exclusive = left.difference(right)
                right_exclusive = right.difference(left)
                msg = "Mismatching column names: "

                if left_exclusive:
                    msg += "Right does not have columns: {}. "

                if right_exclusive:
                    msg += "Left does not have columns: {}. "

                raise AssertionError(msg)

    def _assert_dtypes(self, other: 'TestDataTable'):
        """Check for matching dtypes.

        """

        left_dtypes = {name: column.dtype
                       for name, column in self.parent._columns.items()}

        right_dtypes = {name: column.dtype
                        for name, column in other._columns.items()}

        if left_dtypes != right_dtypes:
            msg = "Mismatching types: "
            for column, left_dtype in left_dtypes.items():
                right_dtype = right_dtypes[column]
                if left_dtype != right_dtype:
                    msg += ("{} (left '{}' vs. right '{}'"
                            .format(column, left_dtype, right_dtype))

            raise AssertionError(msg)

    @staticmethod
    def _get_row_order(table: 'TestDataTable') -> List[int]:
        """Helper function to get index order of sorted data.

        """

        indices = range(table.n_rows)
        return sorted(indices, key=lambda k: table.data[k])


class TestDataTable:
    """Resembles a dataframe in plain python. Its main purpose is to represent
    test data that is independent of any computation engine specific
    characteristics. It serves as a common baseline format. However, in order
    to be usable for all engines, it can be converted to and from any
    computation engine's data representation. This allows to formulate test
    data in an engine independent way only once and to employ it for all
    computation engines simultaneously.

    The main focus lies on simple but correct data representation. This
    includes explicit values for NULL and NaN. Each column needs to be typed.
    Available types are integer, boolean, string, float and datetime.  For
    simplicity, all values will be represented as plain python types
    (no 3rd party). Hence, it is not intended to be used for large amounts of
    data due to its representation in plain python objects.

    There are several limitations. No index column is supported (as in pandas).
    Mixed dtypes are not supported (like dtype object in pandas). No
    distinction is made between int32/int64 or single/double floats. Only
    primitive/atomic types are supported (pyspark's ArrayType or MapType are
    currently not supported).

    Essentially, a test dataframe consists of only 3 attributes: column names,
    column types and column values. In addition, it provides conversion methods
    for all computation engines. It does not offer any computation methods
    itself because it only represents data.

    """

    def __init__(self,
                 data: List[List],
                 columns: List[str],
                 dtypes: List[str]):

        # set attributes
        self.data = data
        self.columns = columns
        self.dtypes = dtypes

        # validate inputs
        self._validata_inputs()

        # convenient attributes
        self.n_rows = len(data)
        self.n_cols = len(columns)

        zipped = zip(columns, dtypes, zip(*data))
        _columns = [(column, TestDataColumn(column, dtype, data))
                    for column, dtype, data in zipped]
        self._columns = collections.OrderedDict(_columns)

        self.assert_equal = EqualityAsserter(self)

    def to_pandas(self) -> pd.DataFrame:
        """Converts test data table into a pandas dataframe.

        """

        data = {name: column.to_pandas()
                for name, column in self._columns.items()}

        return pd.DataFrame(data)

    def to_pyspark(self):
        """Converts test data table into a pandas dataframe.

        """

        from pyspark.sql import SparkSession
        from pyspark.sql import types

        converted = [column.to_pyspark() for column in self._columns.values()]
        fields, values = zip(*converted)

        data = list(zip(*values))
        schema = types.StructType(fields)

        spark = SparkSession.builder.getOrCreate()
        return spark.createDataFrame(data=data, schema=schema)

    def _validata_inputs(self):
        """Check input parameter in regard to validity constraints.

        """

        # assert equal number of elements across rows
        row_lenghts = {len(row) for row in self.data}
        if len(row_lenghts) > 1:
            raise ValueError("Input data has varying number of values per "
                             "row. Please check provided input data")

        # assert equal number of columns and elements per row
        row_lenghts.add(len(self.columns))
        if len(row_lenghts) > 1:
            raise ValueError("Number of columns has to equal the number of "
                             "values per row. Please check column names and "
                             "provided input data.")

        # assert equal number of dtypes and elements per row
        row_lenghts.add(len(self.dtypes))
        if len(row_lenghts) > 1:
            raise ValueError("Number of dtypes has to equal the number of "
                             "values per row. Please check dtypes and "
                             "provided input data.")

        # assert valid dtypes
        for dtype in self.dtypes:
            if dtype not in PRIMITIVE_TYPES:
                raise ValueError("Type '{}' is invalid. Following types are "
                                 "allowed: {}"
                                 .format(dtype, PRIMITIVE_TYPES.keys()))

        # assert unique column names
        duplicates = {x for x in self.columns if self.columns.count(x) > 1}
        if duplicates:
            raise ValueError("Duplicated column names encountered: {}. "
                             "Please use unique column names."
                             .format(duplicates))

    @staticmethod
    def from_pandas(df: pd.DataFrame, dtypes: TYPE_DTYPE_INPUT = None) \
            -> 'TestDataTable':
        """Converts pandas dataframe into TestDataTabble.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to be converted.
        dtypes: list, dict, optional
            If list is provided, each value represents a dtype and maps to
            one column of the dataframe in given order. If dict is provided,
            keys refer to column names and values represent dtypes.

        Returns
        -------
        datatable: TestDataTable
            Converted dataframe

        """

        converter = ConverterFromPandas(df)

        return converter(dtypes=dtypes)

    @staticmethod
    def from_pyspark(df: 'pyspark.sql.DataFrame') -> 'TestDataTable':
        """Converts pandas dataframe into TestDataTabble.

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Dataframe to be converted.

        Returns
        -------
        datatable: TestDataTable
            Converted dataframe

        """

        converter = ConverterFromPySpark(df)

        return converter()

    def __getitem__(self, name: str) -> TestDataColumn:
        """Convenient access to TestDataColumn via column name.

        Parameters
        ----------
        name: str
            Label identifier for columns.

        Returns
        -------
        column: TestDataColumn

        """

        return self._columns[name]

    def __repr__(self):
        """Represent table as ASCII representation.

        """

        headers = ["{}\n({})".format(column, dtype)
                   for column, dtype in zip(self.columns, self.dtypes)]

        preserve = copy.copy(tabulate.MIN_PADDING)
        tabulate.MIN_PADDING = 0

        _repr = tabulate.tabulate(tabular_data=self.data,
                                  headers=headers,
                                  numalign="center",
                                  stralign="center",
                                  tablefmt="psql",
                                  showindex="always")

        tabulate.MIN_PADDING = preserve
        return _repr


def concretize_abstract_wrangler(wrangler_class: Type) -> Type:
    """Makes abstract wrangler classes instantiable for testing purposes by
    implementing abstract methods of `BaseWrangler`.

    Parameters
    ----------
    wrangler_class: Type
        Class object to inherit from while overriding abstract methods.

    Returns
    -------
    concrete_class: Type
        Concrete class usable for testing.

    """

    class ConcreteWrangler(wrangler_class):

        @property
        def preserves_sample_size(self):
            return super().preserves_sample_size

        @property
        def computation_engine(self):
            return super().computation_engine

        def fit(self, *args, **kwargs):
            return super().fit(*args, **kwargs)

        def fit_transform(self, *args, **kwargs):
            return super().fit_transform(*args, **kwargs)

        def transform(self, *args, **kwargs):
            return super().transform(*args, **kwargs)

    ConcreteWrangler.__name__ = wrangler_class.__name__
    ConcreteWrangler.__doc__ = wrangler_class.__doc__

    return ConcreteWrangler
