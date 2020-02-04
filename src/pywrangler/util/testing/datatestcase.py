"""This module contains the DataTestCase class.

"""
from functools import wraps, partial
from typing import Callable, Optional, Dict, Any, Union, List, Sequence

import pandas as pd
from pywrangler.util.testing.mutants import BaseMutant
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

    def __call__(self, test_func: Callable,
                 test_kwargs: Optional[Dict[str, Any]] = None,
                 engine: Optional[str] = None, **kwargs):
        """Assert test data input/output equality for a given test function.
        Input data is passed to the test function and the result is compared
        to output data. Chooses computation engine as specified by parent or
        given by `engine`.

        Parameters
        ----------
        test_func: callable
            A function that takes a pandas dataframe as the first keyword
            argument.
        test_kwargs: dict, optional
            Keyword arguments which will be passed to `test_func`.
        kwargs: dict, optional
            Any computation specific keyword arguments (like `repartition` for
            pyspark).
        engine: str, optional
            Set computation engine to perform test with.

        Raises
        ------
        AssertionError is thrown if computed and expected results do not match.

        """

        engine = engine or self.parent.engine

        if not engine:
            raise ValueError("EngineTester: Computation engine needs to be "
                             "provided either via DataTestCase instantiation "
                             "or via calling `DataTestCase.test()`.")

        engines = {"pandas": self.pandas,
                   "pyspark": self.pyspark}

        asserter = engines.get(engine)
        if not asserter:
            raise ValueError("Provided engine `{}` is not valid. Available "
                             "engines are: {}."
                             .format(engine, engines.keys()))

        asserter(test_func, test_kwargs=test_kwargs, **kwargs)

    def pandas(self, test_func: Callable,
               test_kwargs: Optional[Dict[str, Any]] = None,
               merge_input: Optional[bool] = False,
               force_dtypes: Optional[Dict[str, str]] = None):
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
        test_kwargs: dict, optional
            Keyword arguments which will be passed to `test_func`.
        merge_input: bool, optional
            Merge input dataframe to the computed result of the test function
            (inner join on index).
        force_dtypes: dict, optional
            Enforce specific dtypes for the returned result of the pandas
            test function. This may be necessary due to float casts when NaN
            values are present.

        Raises
        ------
        AssertionError is thrown if computed and expected results do not match.

        """

        output_func = partial(self._pandas_output,
                              merge_input=merge_input,
                              force_dtypes=force_dtypes)

        return self.generic_assert(test_func=test_func,
                                   test_kwargs=test_kwargs,
                                   output_func=output_func)

    def pyspark(self, test_func: Callable,
                test_kwargs: Optional[Dict[str, Any]] = None,
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
        test_args: iterable, optional
            Positional arguments which will be passed to `test_func`.
        test_kwargs: dict, optional
            Keyword arguments which will be passed to `test_func`.
        repartition: int, list, optional
            Repartition input dataframe.

        Raises
        ------
        AssertionError is thrown if computed and expected results do not match.

        """

        output_func = partial(self._pyspark_output, repartition=repartition)

        return self.generic_assert(test_func=test_func,
                                   test_kwargs=test_kwargs,
                                   output_func=output_func)

    def generic_assert(self, test_func: Callable,
                       test_kwargs: Optional[Dict[str, Any]],
                       output_func: Callable):
        """Generic assertion function for all computation engines which
        requires a computation engine specific output generation function.

        Parameters
        ----------
        test_func: callable
            A function that takes a pandas dataframe as the first keyword
            argument.
        test_kwargs: dict, optional
            Keyword arguments which will be passed to `test_func`.
        output_func: callable
            Output generation function which is computation engine specific.

        """

        test_kwargs = test_kwargs or {}
        test_func = partial(test_func, **test_kwargs)

        pf_input = self.parent.input
        pf_output = self.parent.output

        generate_output = partial(output_func,
                                  pf_input=pf_input,
                                  test_func=test_func)

        # standard
        output_computed = generate_output()
        output_computed.assert_equal(pf_output)

        # mutants
        self.generic_assert_mutants(generate_output)

    @staticmethod
    def _pyspark_output(pf_input: PlainFrame, test_func: Callable,
                        repartition: Optional[Union[int, List[str]]],
                        mutant: Optional[BaseMutant] = None) -> PlainFrame:
        """Helper function to generate computed output of DataTestCase for
        given test function.

        Parameters
        ----------
        pf_input: PlainFrame
            Test data input.
        test_func: callable
            A function that takes a pandas dataframe as the first keyword
            argument.
        repartition: int, list, optional
            Repartition input dataframe.
        mutant: BaseMutant, optional
            Optional mutant to modify input dataframe.

        Returns
        -------
        output_computed: PlainFrame

        """

        # check for mutation
        if mutant:
            pf_input = mutant.mutate(pf_input)
        df_input = pf_input.to_pyspark()

        # engine specific
        if repartition is not None:
            df_input = df_input.repartition(repartition)

        # compute result
        df_result = test_func(df_input)
        output_computed = PlainFrame.from_pyspark(df_result)

        return output_computed

    @staticmethod
    def _pandas_output(pf_input: PlainFrame, test_func: Callable,
                       merge_input: Optional[bool],
                       force_dtypes: Optional[Dict[str, str]] = None,
                       mutant: Optional[BaseMutant] = None):
        """Helper function to generate computed output of DataTestCase for
        given test function.

        Parameters
        ----------
        pf_input: PlainFrame
            Test data input.
        test_func: callable
            A function that takes a pyspark dataframe as the first keyword
            argument.
        merge_input: bool, optional
            Merge input dataframe to the computed result of the test function
            (inner join on index).
        mutant: BaseMutant, optional
            Optional mutant to modify input dataframe.
        force_dtypes: dict, optional
            Enforce specific dtypes for the returned result of the pandas
            test function. This may be necessary due to float casts when NaN
            values are present.

        Returns
        -------
        output_computed: PlainFrame

        """

        # check for mutation
        if mutant:
            pf_input = mutant.mutate(pf_input)
        df_input = pf_input.to_pandas()

        # compute result
        df_result = test_func(df_input)

        if merge_input:
            if isinstance(df_result, pd.Series):
                df_result = df_input.assign(**{df_result.name: df_result})
            else:
                df_result = pd.merge(df_input, df_result, left_index=True,
                                     right_index=True, how="inner")

        output_computed = PlainFrame.from_pandas(df_result,
                                                 dtypes=force_dtypes)

        return output_computed

    def generic_assert_mutants(self, func_generate_output: Callable):
        """Given a computation engine specific output generation function
        `generate_output`, iterate all available mutants and confirm their test
        assertion.

        Parameters
        ----------
        func_generate_output: callable
            Computation engine specific function that creates output
            PlainFrame given a mutant.

        Raises
        ------
        AssertionError is raised if a mutant is not killed.

        """

        for mutant in self.parent.mutants:
            output_computed = func_generate_output(mutant=mutant)

            try:
                output_computed.assert_equal(self.parent.output)
                killed = False

            except AssertionError:
                killed = True

            finally:
                if not killed:
                    raise AssertionError("DataTestCase: Mutant {} survived."
                                         .format(mutant))


def convert_method(func: Callable, convert: Callable) -> Callable:
    """Helper function to wrap a given function with a given converter
    function.

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        raw = func(self, *args, **kwargs)
        return convert(raw)

    return wrapper


class TestDataConverter(type):
    """Metaclass for DataTestCase. It's main purpose is to simplify the usage
    of DataTestCase and to avoid boilerplate code.

    Essentially, it wraps and modifies the results of the `input`, `output` and
    `mutants` methods of DataTestCase.

    For `input` and `output`, in converts the result to PlainFrame. For
    `mutants`, it converts the result to BaseMutant. Additionally, methods are
    wrapped as properties for simple dot notation access.

    """

    def __new__(mcl, name, bases, nmspc):
        mandatory = {"input", "output"}.intersection(nmspc.keys())

        wrapped = {key: convert_method(nmspc[key], PlainFrame.from_any)
                   for key in mandatory}

        mutant_func = nmspc.get("mutants", lambda x: [])
        wrapped["mutants"] = convert_method(mutant_func,
                                            BaseMutant.from_multiple_any)

        newclass = super(TestDataConverter, mcl).__new__(mcl, name, bases,
                                                         nmspc)
        for key, value in wrapped.items():
            setattr(newclass, key, property(value))

        return newclass

    def __init__(cls, name, bases, nmspc):
        super(TestDataConverter, cls).__init__(name, bases, nmspc)


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

    In any case, you may also provide a `PlainFrame` directly.

    """

    def __init__(self, engine: Optional[str] = None):
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

        Mutants can be defined in various formats. You can provide a single
        mutant like:
        >>> return ValueMutant(column="col1", row=0, value=3)

        This is identical to the dictionary notation:
        >>> return {("col1", 0): 3}

        If you want to provide multiple mutations within one mutant at once,
        you can use the `MutantCollection` or simply rely on the dictionary
        notation:
        >>> return {("col1", 2): 5, ("col2", 1): "asd"}

        If you want to provide multiple mutants at once, you may provide
        multiple dictionaries within a list:
        >>>  [{("col1", 2): 5}, {("col1", 2): 3}]

        Overall, all subclasses of `BaseMutant` are allowed to be used. You may
        also mix a specialized mutant with the dictionary notation:
        >>> [RandomMutant(), {("col1", 0): 1}]

        """


class TestCollection:
    """Contains one or more DataTestCases. Provides convenient functions to
    be testable as a group (e.g. for pytest).

    Attributes
    ----------
    testcases: List[DataTestCase]
        List of collected DataTestCase instances.
    test_kwargs: dict, optional
        A dict of optional parameter configuration which could be applied to
        collected DataTestCase instances. Keys refer to configuration names.
        Values refer to dicts which in turn represent keyword arguments.

    """

    def __init__(self, datatestcases: Sequence[DataTestCase],
                 test_kwargs: Optional[Dict[str, Dict]] = None):
        self.testcases = datatestcases
        self.test_kwargs = test_kwargs or {}

    @property
    def names(self):
        return [testcase.__name__ for testcase in self.testcases]

    def pytest_parametrize_testcases(self,
                                     arg: Union[str, Callable]) -> Callable:
        """Convenient decorator to wrap a test function which will be
        parametrized with all available DataTestCases in pytest conform manner.

        Decorator can be called before wrapping the test function to supply
        a custom parameter name or can be used directly with the default
        parameter name (testcase). See examples for more.

        Parameters
        ----------
        arg: str, callable
            Name of the argument that will be used within the wrapped test
            function if decorator gets called.

        Examples
        --------

        If not used with a custom parameter name, `testcase` is used by
        default:

        >>> test_collection = TestCollection([test1, test2])
        >>> @test_collection.pytest_parametrize_testcases
        >>> def test_dummy(testcase):
        >>>     testcase().test.pandas(some_func)

        If a custom parameter name is provided, it will be used:

        >>> test_collection = TestCollection([test1, test2])
        >>> @test_collection.pytest_parametrize_testcases("customname")
        >>> def test_dummy(customname):
        >>>     customname().test.pandas(some_func)

        """

        import pytest

        param = dict(argvalues=self.testcases, ids=self.names)

        if isinstance(arg, str):
            param["argnames"] = arg
            return pytest.mark.parametrize(**param)
        else:
            param["argnames"] = "testcase"
            return pytest.mark.parametrize(**param)(arg)

    def pytest_parametrize_kwargs(self, identifier: str) -> Callable:
        """Convenient decorator to access provided `test_kwargs` and wrap them
        into `pytest.mark.parametrize`.

        Parameters
        ----------
        identifier: str
            The name of the test kwargs.


        Examples
        --------

        In the following example, `conf1` represents an available configuration
        to be tested. `param1` and `param2` will be passed to the actual test
        function.

        >>> kwargs= {"conf1": {"param1": 1, "param2": 2}}
        >>> test_collection = TestCollection([test1, test2])
        >>> @test_collection.pytest_parametrize_testcases
        >>> @test_collection.pytest_parametrize_kwargs("conf1")
        >>> def test_dummy(testcase, conf1):
        >>>     testcase().test.pandas(some_func, test_kwargs=conf1)

        """

        import pytest

        if identifier not in self.test_kwargs:
            raise ValueError("Provided test kwargs identifier '{}' does "
                             "not exist. Available test kwargs are: {}."
                             .format(identifier, self.test_kwargs.keys()))

        keys, values = zip(*self.test_kwargs[identifier].items())

        kwargs = dict(argnames=identifier,
                      argvalues=list(values),
                      ids=list(keys))

        return pytest.mark.parametrize(**kwargs)
