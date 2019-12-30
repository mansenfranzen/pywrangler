"""This module contains the DataTestCase class.

"""
from functools import wraps
from typing import Callable, Optional, Iterable, Dict, Any, Union, List

import pandas as pd
from pywrangler.util.testing.plainframe import PlainFrame


class EngineTester:
    """Composite of `DataTestCase` which resembles a collection of engine
    specific assertion functions. More concretely, for each computation engine,
    the input data from the parent data test case is passed to the function to
    be tested. The result is then compared to the output data of the parent
    data test case. Each engine may additionally provide engine specific
    functionality (like repartition for pyspark).

    """

    def __init__(self, parent: 'DataTestCase'):
        self.parent = parent

    def __call__(self, test_func: Callable, args: Optional[Iterable] = None,
                 kwargs: Optional[Dict[str, Any]] = None, **test_kwargs):
        """Assert test data input/output equality for a given test function.
        Input  data is passed to the test function and the result is compared
        to output data. Chooses computation engine as specified by parent.

                Parameters
        ----------
        test_func: callable
            A function that takes a pandas dataframe as the first keyword
            argument.
        args: iterable, optional
            Positional arguments which will be passed to `func`.
        kwargs: dict, optional
            Keyword arguments which will be passed to `func`.
        test_kwargs: dict, optional
            Any computation specific keyword arguments (like `repartition` for
            pyspark).

        """

        engine = {"pandas": self.pandas,
                  "pyspark": self.pyspark}

        engine[self.parent.engine](test_func,
                                   args=args,
                                   kwargs=kwargs,
                                   **test_kwargs)

    def pandas(self, test_func: Callable, args: Optional[Iterable] = None,
               kwargs: Optional[Dict[str, Any]] = None,
               merge_input: Optional[bool] = False):
        """Assert test data input/output equality for a given test function.
        Input  data is passed to the test function and the result is compared
        to output data.

        Some data test cases require the test function to add new columns
        to the input dataframe where correct row order is mandatory. In
        those cases, pandas test functions may only return new columns
        instead of adding columns to the input dataframe (modifying the
        input dataframe may result in performance penalties and hence
        should be prevented). This is special to pandas since it provides
        an index containing the row order information and does not require
        the input dataframe to be modified. However, data test cases are
        formulated to include the input dataframe within the output
        dataframe when row order matters because other engines may not have
        an explicit index column (e.g. pyspark). To account for this pandas
        specific behaviour, `merge_input` can be activated to make the
        assertion behave appropriately.

        Parameters
        ----------
        test_func: callable
            A function that takes a pandas dataframe as the first keyword
            argument.
        args: iterable, optional
            Positional arguments which will be passed to `func`.
        kwargs: dict, optional
            Keyword arguments which will be passed to `func`.
        merge_input: bool, optional
            Merge input dataframe to the computed result of the test function
            (inner join on index).

        """

        args = args or ()
        kwargs = kwargs or {}

        df_input = self.parent.input.to_pandas()
        df_result = test_func(df_input, *args, **kwargs)

        if merge_input:
            if isinstance(df_result, pd.Series):
                df_result = df_input.assign(**{df_result.name: df_result})
            else:
                df_result = pd.merge(df_input, df_result, left_index=True,
                                     right_index=True, how="inner")

        output = self.parent.output
        output.assert_equal(PlainFrame.from_pandas(df_result))

    def pyspark(self, test_func: Callable, args: Optional[Iterable] = None,
                kwargs: Optional[Dict[str, Any]] = None,
                repartition: Optional[Union[int, List[str]]] = None):
        """Assert test data input/output equality for a given test function.
        Input  data is passed to the test function and the result is compared
        to output data.

        Pyspark's partitioning may be explicitly varied to test against
        different partitioning settings via `repartition`.

        Parameters
        ----------
        test_func: callable
            A function that takes a pandas dataframe as the first keyword
            argument.
        args: iterable, optional
            Positional arguments which will be passed to `func`.
        kwargs: dict, optional
            Keyword arguments which will be passed to `func`.
        repartition: int, list, optional
            Repartition input dataframe.

        """

        args = args or ()
        kwargs = kwargs or {}

        df_input = self.parent.input.to_pyspark()

        if repartition is not None:
            df_input = df_input.repartition(repartition)

        df_result = test_func(df_input, *args, **kwargs)

        output = self.parent.output
        output.assert_equal(PlainFrame.from_pyspark(df_result))


class TestDataConverter(type):
    def __new__(mcl, name, bases, nmspc):
        mandatory = ("input", "output")

        for mand in mandatory:
            if mand not in nmspc:
                raise NotImplementedError("DataTestCase '{}' needs to "
                                          "implement '{}' method."
                                          .format(name, mand))

        wrapped = {key: TestDataConverter.ensure_format(nmspc[key])
                   for key in mandatory}

        newclass = super(TestDataConverter, mcl).__new__(mcl, name, bases,
                                                         nmspc)
        for key, func in wrapped.items():
            setattr(newclass, key, property(func))

        return newclass

    def __init__(cls, name, bases, nmspc):
        super(TestDataConverter, cls).__init__(name, bases, nmspc)

    @staticmethod
    def ensure_format(data_func):
        """Helper function to ensure provided data input is correctly converted
        to `PlainFrame`.

        Checks following scenarios: If PlainFrame is given, simply pass. If
        dict is given, call constructor from dict. If tuple is given, pass to
        normal init of PlainFrame.

        """
        print("Wrapping")

        @wraps(data_func)
        def wrapper(self, *args, **kwargs):
            print(args, kwargs)
            result = data_func(self, *args, **kwargs)
            if isinstance(result, PlainFrame):
                return result
            elif isinstance(result, dict):
                return PlainFrame.from_dict(result)
            elif isinstance(result, tuple):
                return PlainFrame(*result)
            else:
                raise ValueError("Unsupported data encountered. Data needs "
                                 "needs to be a TestDataFrame, a dict or a "
                                 "tuple. Provided type is {}."
                                 .format(type(result)))

        return wrapper


class DataTestCase(metaclass=TestDataConverter):
    """Represents a data focused test case which has 3 major goals. First, it
    aims to unify and standardize test data formulation across different
    computation engines. Second, test data should be as readable as possible
    and should be maintainable in pure python. Third, it intends to make
    writing data centric tests as easy as possible while reducing the need of
    test case related boilerplate code.

    To accomplish these goals, (1) it provides an abstraction layer for a
    computation engine independent data representation via `PlainFrame`. Test
    data is formulated once and automatically converted into the target
    computation engine representation. To ensure readability (2), test data may
    be formulated in column or row format with pure python objects. To reduce
    boilerplate code (3), it provides automatic assertion test functionality
    for all  computation engines via `EngineAsserter`. Additionally, it allows
    to define mutants of the input data which should cause the test to fail
    (hence covering multiple distinct but similar test data scenarios within
    the same data test case).

    Every data test case implements `input` and `output` methods. They resemble
    the data given to a test function and the computed data expected from the
    corresponding test function, respectively. Since the data needs to be
    formulated in a computation engine independent format, the `PlainFrame` is
    is used. For convenience, there are multiple ways of instantiation of a
    `PlainFrame` as a dict or tuple.

    A dict requires typed column names as keys and values as values, which
    resembles the column format (define values column wise):
    >>> result = {"col1:int": [1,2,3], "col2:str": ["a", "b", "c"]}

    A tuple may be returned in 2 variants. Both represent the row format
    (define values row wise). The most verbose way is to include data, column
    names and dtypes.
    >>> data = [[1, "a"],
    >>>         [2, "b"],
    >>>         [3, "b"]]
    >>> columns = ["col1", "col2"]
    >>> dtypes = ["int", "str"]
    >>> result = (data, columns, dtypes)

    Second, dtypes may be provided simultaneously with column names as
    typed column annotations:
    >>> data = [[1, "a"], [2, "b"], [3, "b"]]
    >>> columns = ["col1:int", "col2:str"]
    >>> result = (data, columns)

    In any case, you may also provide `PlainFrame` directly.

    """

    def __init__(self, engine):
        self.engine = engine
        self.test = EngineTester(self)

    def input(self):
        """Represents the data input given to a data transformation function
        to be tested.

        It needs to be implemented by every data test case.

        """

        raise NotImplementedError

    def output(self):
        """Represents the data output expected from data transformation
        function to be tested.

        It needs to be implemented by every data test case.

        """

        raise NotImplementedError

    def mutants(self):
        """Mutants describe modifications to the input data which should cause
        the test to fail.

        """

        ("col1", 3)