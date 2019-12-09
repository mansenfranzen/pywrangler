"""This module contains testing utility.

"""
import numbers
from datetime import datetime
from typing import Iterable, Tuple, Type

import dateutil
import numpy as np
import pandas as pd


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
    """Represents null values.

    """

    def __str__(self):
        return "NULL"


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
    all values will be represented as plain python types (no 3rd party).

    Essentially, a test dataframe consists of only 3 attributes: column names,
    column types and column values. In addition, it provides conversion methods
    for all computation engines. It does not offer any computation methods
    itself because it only represents data.

    """

    def __init__(self,
                 data: Iterable[Iterable],
                 columns: Iterable[str],
                 dtypes: Iterable[str]):

        # assert equal number of elements across rows
        row_lenghts = {len(row) for row in data}
        if len(row_lenghts) > 1:
            raise ValueError("Input data has varying number of values per "
                             "row. Please check provided input data")

        # assert equal number of columns and elements per row
        row_lenghts.add(len(columns))
        if len(row_lenghts) > 1:
            raise ValueError("Number of columns has to equal the number of "
                             "values per row. Please check column names and "
                             "provided input data.")

        # assert equal number of dtypes and elements per row
        row_lenghts.add(len(dtypes))
        if len(row_lenghts) > 1:
            raise ValueError("Number of dtypes has to equal the number of "
                             "values per row. Please check dtypes and "
                             "provided input data.")

        # assert valid dtypes
        for dtype in dtypes:
            if dtype not in TYPES:
                raise ValueError("Type '{}' is invalid. Following types are "
                                 "allowed: {}"
                                 .format(dtype, TYPES.keys()))

        zipped = zip(columns, dtypes, zip(*data))
        self._columns = [TestDataColumn(column, dtype, data)
                         for column, dtype, data in zipped]

    def to_pandas(self) -> pd.DataFrame:
        """Converts test data table into a pandas dataframe.

        """

        data = {column.name: column.to_pandas() for column in self._columns}
        return pd.DataFrame(data)

    def to_pyspark(self):
        """Converts test data table into a pandas dataframe.

        """

        from pyspark.sql import SparkSession
        from pyspark.sql import types

        converted = [column.to_pyspark() for column in self._columns]
        fields, values = zip(*converted)

        data = list(zip(*values))
        schema = types.StructType(fields)

        spark = SparkSession.builder.getOrCreate()
        return spark.createDataFrame(data=data, schema=schema)
