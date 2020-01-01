"""This module contains the PlainFrame and PlainColumn classes.

"""
import collections
import copy
import functools
import numbers
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tabulate
from numpy.testing import assert_equal
from pandas.api import types


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

TYPE_ABBR = {"i": "int",
             "b": "bool",
             "f": "float",
             "s": "str",
             "d": "datetime"}


class PlainColumn:
    """Represents a single column of a PlainFrame. Handles dtype specific
    preprocessing and performs type validation. In addition, it contains
    conversion methods for all supported computation engines.

    """

    def __init__(self, name: str, dtype: str, values: Sequence):

        self.name = name
        self.dtype = dtype

        # preprocess
        if dtype == "float":
            values = self._preprocess_float(values)
        elif dtype == "datetime":
            values = self._preprocess_datetime(values)

        # sanity check for dtypes
        self.values = values
        self._check_dtype()

        # get null/nan flags
        self.has_null = any([x is NULL for x in values])
        self.has_nan = any([x is np.NaN for x in values])

        # add composite converters
        self.to_pandas = ConverterToPandas(self)
        self.to_pyspark = ConverterToPySpark(self)

    @staticmethod
    def _preprocess_datetime(values: Sequence) \
            -> Tuple[Union[datetime, NullValue]]:
        """Convenience method to allow timestamps of various formats.

        """

        processed = [pd.Timestamp(x).to_pydatetime()
                     if not isinstance(x, NullValue)
                     else x
                     for x in values]

        return tuple(processed)

    @staticmethod
    def _preprocess_float(values: Sequence) -> Tuple[Union[float, NullValue]]:
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
                raise TypeError("Column '{}' has invalud value '{}' with "
                                "invalid type '{}'. Allowed types are: {}."
                                .format(self.name,
                                        value,
                                        type(value),
                                        allowed_types))

    def modify(self, modifications: Dict[int, Any]) -> 'PlainColumn':
        """Modifies PlainColumn and return new instance.

        """

        n_rows = len(self.values)
        values = [modifications.get(idx, self.values[idx])
                  for idx in range(n_rows)]

        return PlainColumn(name=self.name, dtype=self.dtype, values=values)


class PlainFrame:
    """Resembles a dataframe in plain python. Its main purpose is to represent
    test data that is independent of any computation engine specific
    characteristics. It serves as a common baseline format. However, in order
    to be usable for all engines, it can be converted to and from any
    computation engine's data representation. This allows to formulate test
    data in an engine independent way only once and to employ it for all
    computation engines simultaneously.

    The main focus lies on simple but correct data representation. This
    includes explicit values for NULL and NaN. Each column needs to be typed.
    Available types are integer, boolean, string, float and datetime. For
    simplicity, all values will be represented as plain python types
    (no 3rd party). Hence, it is not intended to be used for large amounts of
    data due to its representation in plain python objects.

    There are several limitations. No index column is supported (as in pandas).
    Mixed dtypes are not supported (like dtype object in pandas). No
    distinction is made between int32/int64 or single/double floats. Only
    primitive/atomic types are supported (pyspark's ArrayType or MapType are
    currently not supported).

    Essentially, a PlainFrame consists of only 3 attributes: column names,
    column types and column values. In addition, it provides conversion methods
    for all computation engines. It does not offer any computation methods
    itself because it only represents data.

    """

    def __init__(self,
                 data: Sequence[Sequence],
                 columns: Sequence[str],
                 dtypes: Optional[Sequence[str]] = None):
        """Initialize `PlainFrame`. Dtypes have to provided either via
        `columns` as typed column annotations or directly via `dtypes`.
        Typed column annotations are a convenient way to omit the `dtypes`
        parameter while specifying dtypes directly with the `columns`
        parameter.

        An exmaple of a typed column annotation is as follows:
        >>> columns = ["col_a:int", "col_b:str", "col_c:float"]

        Abbreviations may also be used like:
        >>> columns = ["col_a:i", "col_b:s", "col_c:f"]

        For a complete abbreviation mapping, please see `TYPE_ABBR`.

        Parameters
        ----------
        data: list
            List of iterables representing the input data.
        columns: list
            List of strings representing the column names. Typed annotations
            are allowed to be used here and will be checked of `dtypes` is not
            provided.
        dtypes: list, optional
            List of column types.

        """

        # set attributes
        self.data = data

        # check for typed columns
        if dtypes is None:
            self.columns, self.dtypes = self._parse_typed_columns(columns)
        else:
            self.columns = columns
            self.dtypes = dtypes

        # validate inputs
        self._validata_inputs()

    @property
    def n_rows(self) -> int:
        """Return the number of rows.

        """

        return len(self.data)

    @property
    def n_cols(self):
        """Returns the number columns.

        """

        return len(self.columns)

    @property
    @functools.lru_cache()
    def plaincolumns(self) -> 'OrderedDict[str, PlainColumn]':
        """Creates an ordered dictionary of PlainColumn instances. Keys refer
        to column names and values represent actual PlainColumn instances. This
        mainly used for column wise operations.

        """

        zipped = zip(self.columns, self.dtypes, zip(*self.data))
        columns = [(column, PlainColumn(column, dtype, data))
                   for column, dtype, data in zipped]

        return OrderedDict(columns)

    @property
    @functools.lru_cache()
    def assert_equal(self) -> 'EqualityAsserter':
        """Return equality assertion composite.

        """

        return EqualityAsserter(self)

    def modify(self, modifications: Dict[str, Dict[int, Any]]):
        """Modifies PlainFrame with given modifications and return new instance
        of it.

        """

        columns = OrderedDict()

        for name, column in self.plaincolumns.items():
            if name in modifications:
                modification = modifications[name]
                modified = column.modify(modification)
                columns[name] = modified
            else:
                columns[name] = column

        return PlainFrame.from_plaincolumns(columns)

    @classmethod
    def from_plaincolumns(cls, plaincolumns: 'OrderedDict[str, PlainColumn]'):

        columns = list(plaincolumns.keys())
        dtypes = [column.dtype for column in plaincolumns.values()]
        data = [column.values for column in plaincolumns.values()]
        data = list(zip(*data))

        return cls(data=data, columns=columns, dtypes=dtypes)

    def to_plaincolumns(self) -> 'OrderedDict[str, PlainColumn]':

        return self.plaincolumns

    def to_pandas(self) -> pd.DataFrame:
        """Converts test data table into a pandas dataframe.

        """

        data = {name: column.to_pandas()
                for name, column in self.plaincolumns.items()}

        return pd.DataFrame(data)

    def to_pyspark(self):
        """Converts test data table into a pandas dataframe.

        """

        from pyspark.sql import SparkSession
        from pyspark.sql import types

        converted = [column.to_pyspark() for column in
                     self.plaincolumns.values()]
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

        # create plaincolumns which has additional type validity checks
        plaincolumns = self.plaincolumns

    @staticmethod
    def from_pandas(df: pd.DataFrame, dtypes: TYPE_DTYPE_INPUT = None) \
            -> 'PlainFrame':
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
        datatable: PlainFrame
            Converted dataframe

        """

        converter = ConverterFromPandas(df)

        return converter(dtypes=dtypes)

    @staticmethod
    def from_pyspark(df: 'pyspark.sql.DataFrame') -> 'PlainFrame':
        """Converts pandas dataframe into TestDataTabble.

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Dataframe to be converted.

        Returns
        -------
        datatable: PlainFrame
            Converted dataframe

        """

        converter = ConverterFromPySpark(df)

        return converter()

    @staticmethod
    def from_dict(data: 'collections.OrderedDict[str, Sequence]') \
            -> 'PlainFrame':
        """Converts a python dict into PlainFrame. Assumes keys to be column
        names with type annotations and values to be values.

        Parameters
        ----------
        data: dict
            Keys represent typed column annotations and values represent data
            values.

        Returns
        -------
        datatable: PlainFrame

        """

        columns, values = zip(*data.items())
        values = list(zip(*values))

        return PlainFrame(data=values, columns=columns)

    def to_dict(self) -> 'collections.OrderedDict[str, tuple]':
        """Converts PlainFrame into dictionary with key as typed columns
        and values as data.

        Returns
        -------
        table_dict: OrderedDict

        """

        columns = [("{}:{}".format(column.name, column.dtype), column.values)
                   for column in self.plaincolumns.values()]

        return collections.OrderedDict(columns)

    @staticmethod
    def _parse_typed_columns(typed_columns: Sequence[str]) \
            -> Tuple[List[str], List[str]]:
        """Separates column names and corresponding type annotations from
        column names with type annotation strings.

        For example, ["a:int", "b:str"] will be separated into ("a", "b"),
        ("int", "str").

        """

        splitted = [x.split(":") for x in typed_columns]

        # assert correct split
        invalid = [x for x in splitted if len(x) != 2]
        if invalid:
            raise ValueError("Invalid typed column format encountered: {}. "
                             "Typed columns should be formulated like "
                             "'col_name:type_name', e.g. 'col1:int'. Please "
                             "be aware that this error may occur if you omit "
                             "dtypes when instantiating `PlainFrame`."
                             .format(invalid))

        # get column names and corresponding types
        cols, types = zip(*splitted)

        # complete type abbreviations
        types = [TYPE_ABBR.get(x, x) for x in types]

        # check valid types
        invalid_types = set(types).difference(TYPE_ABBR.values())
        if invalid_types:
            raise ValueError("Invalid types encountered: {}. Valid types "
                             "are: {}."
                             .format(invalid_types, TYPE_ABBR.items()))

        return cols, types

    def get_column(self, name: str) -> PlainColumn:
        """Convenient access to TestDataColumn via column name.

        Parameters
        ----------
        name: str
            Label identifier for columns.

        Returns
        -------
        column: PlainColumn

        """

        return self.plaincolumns[name]

    def __getitem__(self, subset: Union[str, Sequence[str], slice]) \
            -> 'PlainFrame':
        """Get labeled based subset of PlainFrame. Supports single columns,
        list and slices of columns.

        Parameters
        ----------
        columns: str, list, slice
            Defines column subset to be returned. If single str is passed,
            returns single column. If list of strings is passed, returns
            corresponding columns. If slice is passed, returns all columns
            included within slice (start and end including).

        Returns
        -------
        table: PlainFrame

        """

        # handle different input types
        if isinstance(subset, str):
            columns = [subset]
        elif isinstance(subset, (list, tuple)):
            columns = subset
        elif isinstance(subset, slice):
            start = subset.start
            stop = subset.stop

            idx_start = self.columns.index(start)
            idx_stop = self.columns.index(stop)

            columns = self.columns[idx_start:idx_stop + 1]
        else:
            raise ValueError("Subsetting requires str, list, tuple or slice. "
                             "However, {} was encountered."
                             .format(type(subset)))

        # check column names
        invalid = [column for column in columns
                   if column not in self.columns]

        if invalid:
            raise ValueError("Columns '{}' does not exist. Available column "
                             "names are: {}"
                             .format(invalid, self.columns))

        # get dtypes and data
        dtypes = [self.get_column(column).dtype
                  for column in columns]

        data = [self.get_column(column).values
                for column in columns]
        # transpose data
        data = list(zip(*data))

        return PlainFrame(data=data, columns=columns, dtypes=dtypes)

    def __repr__(self):
        """Get table as ASCII representation.

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


class ConverterFromPySpark:
    """Convert pyspark dataframe into PlainFrame.

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

    def __call__(self) -> 'PlainFrame':
        """Converts pyspark dataframe to PlainFrame. Several types are not
        supported including BinaryType, DecimalType, ByteType, ArrayType and
        MapType.

        Returns
        -------
        datatable: pywrangler.util.testing.plainframe.PlainFrame
            Converted dataframe.

        """

        data = list(map(self.convert_null, self.df.collect()))
        columns, dtypes = self.get_column_dtypes()

        return PlainFrame(data=data, columns=columns, dtypes=dtypes)

    def get_column_dtypes(self) -> Tuple[List[str], List[str]]:
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
    PlainColumn. It handles spark specifics like NULL as None and proper
    type matching.

    """

    def __init__(self, parent: 'PlainColumn'):
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
    """Convert pandas dataframe into plain PlainFrame.

    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self,
                 dtypes: Optional[TYPE_DTYPE_INPUT] = None) \
            -> 'PlainFrame':
        """Converts pandas dataframe to PlainFrame. Dtypes will be inferred
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
        datatable: PlainFrame
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

        return PlainFrame(data=data,
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
        """Converts a column of pandas dataframe into PlainFrame readable
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
    PlainColumn. It handles pandas specifics likes the missing distinction
    between NULL and NaN.

    """

    def __init__(self, parent: 'PlainColumn'):
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


class EqualityAsserter:
    """Collection of equality assertions as a composite of PlainFrame. It
    contains equality tests in regard to number of rows, columns, dtypes etc.

    """

    def __init__(self, parent: 'PlainFrame'):
        self.parent = parent

    def __call__(self,
                 other: 'PlainFrame',
                 assert_column_order: bool = False,
                 assert_row_order: bool = False):
        """Main entry point for equality assertion. By default, no strict
        column nor row order is assumed but may be enabled.

        Parameters
        ----------
        other: PlainFrame
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
            left = self.parent.plaincolumns[column].values
            right = other.plaincolumns[column].values

            if not assert_row_order:
                left = [left[idx] for idx in order_left]
                right = [right[idx] for idx in order_right]

            msg = "column=" + column
            assert_equal(left, right, err_msg=msg)

    def _assert_shape(self, other: 'PlainFrame'):
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
                             other: 'PlainFrame',
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

    def _assert_dtypes(self, other: 'PlainFrame'):
        """Check for matching dtypes.

        """

        left_dtypes = {name: column.dtype
                       for name, column in self.parent.plaincolumns.items()}

        right_dtypes = {name: column.dtype
                        for name, column in other.plaincolumns.items()}

        if left_dtypes != right_dtypes:
            msg = "Mismatching types: "
            for column, left_dtype in left_dtypes.items():
                right_dtype = right_dtypes[column]
                if left_dtype != right_dtype:
                    msg += ("{} (left '{}' vs. right '{}'"
                            .format(column, left_dtype, right_dtype))

            raise AssertionError(msg)

    @staticmethod
    def _get_row_order(table: 'PlainFrame') -> List[int]:
        """Helper function to get index order of sorted data.

        """

        indices = range(table.n_rows)
        return sorted(indices, key=lambda k: table.data[k])
