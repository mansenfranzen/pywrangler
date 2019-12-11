"""This module contains testing utility.

"""
import collections
import numbers
from datetime import datetime
from typing import Iterable, List, Tuple, Type

import dateutil
import numpy as np
import pandas as pd
from numpy.testing import assert_equal


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

TYPES = {"bool": (bool, NullValue),
         "int": (int, NullValue),
         "float": (float, int, NullValue),
         "str": (str, NullValue),
         "datetime": (datetime, NullValue)}


class ConverterPySpark:
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


class ConverterPandas:
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

    def __init__(self, name: str, dtype: str, values: Iterable):

        self.name = name
        self.dtype = dtype
        self.values = values

        # get null/nan flags
        self.has_null = any([x is NULL for x in values])
        self.has_nan = any([x is np.NaN for x in values])

        # preprocess
        if dtype == "float":
            self._preprocess_float()
        elif dtype == "datetime":
            self._preprocess_datetime()

        # sanity check for dtypes
        self._check_dtype()

        # add composite converters
        self.to_pandas = ConverterPandas(self)
        self.to_pyspark = ConverterPySpark(self)

    def _preprocess_datetime(self):
        """Convenience method to allow timestamps to be of type string.
        Converts known timestamp string format (dateutil) to datetime objects.

        """

        self.values = [dateutil.parser.parse(x) if isinstance(x, str) else x
                       for x in self.values]

    def _preprocess_float(self):
        """Convenience method to ensure numeric values are casted to float.

        """

        self.values = [float(x) if isinstance(x, numbers.Number) else x
                       for x in self.values]

    def _check_dtype(self):
        """Ensures correct type of all values. Raises TypeError.

        """

        allowed_types = TYPES[self.dtype]

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

            assert_equal(left, right)

    def _assert_shape(self, other: 'TestDataTable'):
        """Check for identical shape

        """

        assert self.parent.n_rows == other.n_rows
        assert self.parent.n_cols == other.n_cols

    def _assert_column_names(self,
                             other: 'TestDataTable',
                             assert_column_order: bool):
        """Check for matching column names. Take column order into account if
        required.

        """

        if assert_column_order:
            assert self.parent.columns == other.columns
        else:
            assert set(self.parent.columns) == set(other.columns)

    def _assert_dtypes(self, other: 'TestDataTable'):
        """Check for matching dtypes.

        """

        left_dtypes = {name: column.dtype
                       for name, column in self.parent._columns.items()}

        right_dtypes = {name: column.dtype
                        for name, column in other._columns.items()}

        assert left_dtypes == right_dtypes

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

    The main focus lies on correct data representation. This includes explicit
    values for NULL and NaN. Each column needs to be typed. For simplicity,
    all values will be represented as plain python types (no 3rd party). It is
    not intended to be used for lots of data due to its representation in plain
    python objects.

    Essentially, a test dataframe consists of only 3 attributes: column names,
    column types and column values. In addition, it provides conversion methods
    for all computation engines. It does not offer any computation methods
    itself because it only represents data.

    """

    def __init__(self,
                 data: Iterable[Iterable],
                 columns: Iterable[str],
                 dtypes: Iterable[str]):

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
            if dtype not in TYPES:
                raise ValueError("Type '{}' is invalid. Following types are "
                                 "allowed: {}"
                                 .format(dtype, TYPES.keys()))

        # assert unique column names
        duplicates = {x for x in self.columns if self.columns.count(x) > 1}
        if duplicates:
            raise ValueError("Duplicated column names encountered: {}. "
                             "Please use unique column names."
                             .format(duplicates))
