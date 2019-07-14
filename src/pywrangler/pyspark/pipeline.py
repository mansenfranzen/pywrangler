"""This module adds extended pipeline functionality for pyspark.

"""

import inspect
import re
from collections import OrderedDict

import pandas as pd
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import DataFrame

from pywrangler.util._pprint import pretty_time_duration as fmt_time
from pywrangler.util._pprint import textwrap_docstring, truncate

from typing import Dict, Callable, Optional, Sequence, Union, Any

TYPE_STAGE = Union[Transformer, Any]

REGEX_STAGE = re.compile(r".*?\*\((\d+)\).*")

DESC_H = "| ({idx}) - {identifier}, {cols} columns, stage {stage}, {cached}"
DESC_B = "| {text:76} |"
DESC_L = "+" + "-" * 78 + "+"
DESC_A = " " * 38 + "||\n" + " " * 37 + "\\||/\n" + " " * 38 + r"\/"

PROF_H = "| Idx |    Identifier    | Total time | Partial time " \
         "| Cached time | Stage | Shape (rows x cols) |"
PROF_L = "+-----+------------------+------------+--------------" \
         "+-------------+-------+---------------------+"
PROF_B = "| {idx:^3} | {name:^16} | {total_time:^10} | {partial_time:^12} " \
         "| {cached_time:^11} | {stage:^5} | {shape:^19} |"

ERR_TYPE_ACCESS = "Value has incorrect type '{}' (integer or string allowed)."


def _create_getter_setter(name: str) -> Dict[str, Callable]:
    """Helper function to create getter and setter methods
    for parameters of `Transformer` class for given parameter
    name.

    """

    def setter(self, value):
        return self._set(**{name: value})

    def getter(self):
        return self.getOrDefault(getattr(self, name))

    return {"get{name}".format(name=name): getter,
            "set{name}".format(name=name): setter}


def func_to_spark_transformer(func: Callable):
    """Convert a native python function into a pyspark `Transformer`
    instance.

    Temporarely creates a new sublcass of type `Transformer` during
    runtime while ensuring that all keyword arguments of the input
    function are mapped to corresponding `Param` values with
    required getter and setter methods for the resulting `Transformer`
    class.

    Returns an instance of the temporarely create `Transformer` subclass.

    """

    class_name = func.__name__
    class_bases = (Transformer,)
    class_dict = {}

    # overwrite transform method while taking care of kwargs
    def _transform(self, df):
        return func(df, **self.getParams())

    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def getParams(self):
        params = self.extractParamMap().items()
        kwargs = {key.name: value for key, value in params}
        return kwargs

    class_dict["_transform"] = _transform
    class_dict["setParams"] = setParams
    class_dict["getParams"] = getParams
    class_dict["__doc__"] = func.__doc__

    # get keyword arguments
    signature = inspect.signature(func)
    parameters = signature.parameters.values()
    parameters = {x.name: x.default for x in parameters
                  if not x.default == inspect._empty}

    # create setter/getter and Param instances
    for parameter in parameters.keys():
        class_dict.update(_create_getter_setter(parameter))
        class_dict[parameter] = Param(Params._dummy(), parameter, "")

    # create class
    transformer_class = type(class_name, class_bases, class_dict)
    transformer_instance = transformer_class()
    transformer_instance._set(**parameters)

    return transformer_instance


class Pipeline(PipelineModel):
    """Represents a compiled pipeline with transformers and fitted models.

    This subclass of `PipelineModel` adds several convenient features to
    extend its usage for normal ETL pipelines. More specifically, it
    enables the user to add native python functions that resemble a valid
    data transformation step while automatically converting python functions
    into valid `Transformer` instances. Transformer functions only require
    to have the first parameter represent the input dataframe. Keyword
    arguments will be available as normal parameters of the resulting pyspark
    `Transformer`.

    In addition, the `_transform` method checks for an optionally defined
    `IsCached` parameter to cache intermediate results for each stage. This
    enables the possibility to cache specific stages. The `describe` method
    gives a brief overview of the stages. The `profile` method allows to get
    detailed timings and provides indicators for possible caching benefits for
    each stage.

    Also, `__call__` is implemented to conveniently access dataframe
    representations for each stage and `__getitem__` allows to access
    stage representation.

    Each pipeline instance may be provided with an explicit documentation
    string.

    Parameters
    ----------
    stages: iterable
        Contains the stages for the pipleline.
    doc: str, optional
        Provide optional doc string for the pipeline.

    """

    def __init__(self, stages: Sequence, doc: Optional[str]=None):
        """Instantiate pipeline. Convert functions into `Transformer`
        instances if necessary.

        In addition to `self.stages`, `self._stage_mapping` allows label based
        access to stages via identifiers, `self._stage_results` keeps track
        of intermediate dataframe representation once `transform` is called
        and `self._stage_profiles` keeps track of profiling results once
        `profile` is called.



        """

        stages = [self._check_convert_transformer(stage) for stage in stages]
        super().__init__(stages)

        self._stage_mapping = OrderedDict()
        self._stage_results = OrderedDict()
        self._stage_profiles = OrderedDict()

        for stage in self.stages:
            identifier = self._create_stage_identifier(stage)
            self._stage_mapping[identifier] = stage

        # overwrite class doc string if pipe doc is explicitly provided
        if doc:
            self.__doc__ = doc

    def _transform(self, df: DataFrame) -> DataFrame:
        """Apply stage's `transform` methods in order while storing
        intermediate stage results and respecting optional cache settings.

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            The input dataframe to be transformed.

        Returns
        -------
        df: pyspark.sql.DataFrame
            The final dataframe representation once all stage transformations
            habe been applied.

        """

        self._stage_results.clear()
        self._stage_results["input_dataframe"] = df

        for identifier, stage in self._stage_mapping.items():
            df = stage.transform(df)

            if self._is_cached(stage):
                df.cache()

            self._stage_results[identifier] = df

        return df

    def describe(self) -> None:
        """Print description of pipeline while reporting index, identifier,
        number of columns, execution plan stage, cache and doc string of every
        stage in order.

        It make some time depending on DAG complexity because `explain` will be
        called on each stage's dataframe representation.

        """

        enumeration = enumerate(self._stage_mapping.items())
        for idx, (identifier, stage) in enumeration:

            # variables
            df = self(identifier)
            cols = len(df.columns)
            exec_stage = self._get_execution_stage(df)
            cached = "cached" if self._is_cached(stage) else "not cached"

            # header string
            header = DESC_H.format(idx=idx,
                                   identifier=truncate(identifier, 35),
                                   cols=cols,
                                   stage=exec_stage,
                                   cached=cached).ljust(79, " ") + "|"

            # doc string
            docs = textwrap_docstring(stage)
            if docs:
                docs = [DESC_B.format(text=text) for text in docs] + [DESC_L]

            # arrow for all but first stage
            if idx != 0:
                print(DESC_A)

            # print description
            print(DESC_L, header, DESC_L, *docs, sep="\n")

    def profile(self, verbose: bool=False) -> None:
        """Profiles each stage of the pipeline with and without caching
        enabled. Total, partial and cache times are reported. Partial time is
        computed as the current total time minus the previous total time. If
        partial and cache time differ greatly, this may indicate multiple
        computations of previous stages and a possible benefit of caching. The
        shape (number of columns and rows) is also reported.

        Before and after profiling, all dataframes are unpersisted. Finally,
        caching is enabled for stages with caching attribute.

        Please be aware that this method will take a while depending on your
        input data because it will call an action on each stage twice and it
        will cache each stage's result temporarily.

        Parameters
        ----------
        verbose: bool, optional
            Enable verbose information about current state.

        Returns
        -------
        None, but prints profile report.

        """

        self._is_transformed()
        self._stage_profiles.clear()

        self._unpersist_dataframes(verbose)
        self._profile_without_caching(verbose)
        self._profile_with_caching(verbose)
        self._unpersist_dataframes(verbose)

        self._restore_caching()
        self._profile_report()

    def _unpersist_dataframes(self, verbose=None):
        """Unpersists all dataframe representations except input dataframe.

        Parameters
        ----------
        verbose: bool, optional
            Enable verbose information about current state.

        """

        if verbose:
            print("Unpersisting all dataframes.")

        for identifier, df in self._stage_results.items():
            if identifier == "input_dataframe":
                continue

            df.unpersist(blocking=True)

    def _profile_without_caching(self, verbose: bool=None) -> None:
        """Make profile without caching enabled. Store total and partial time,
        counts and execution plan stage.

        Parameters
        ----------
        verbose: bool, optional
            Enable verbose information about current state.

        """

        if verbose:
            print("Profile without caching:\n\tProfile input dataframe")

        # profile initial stage
        df = self._stage_results["input_dataframe"]
        prof = self._profile_stage(df)
        prof["partial_time"] = 0
        prof["cached_time"] = 0
        self._stage_profiles["input_dataframe"] = prof

        # keep previous time
        temp_total_time = prof["total_time"]

        for identifier, stage in self._stage_mapping.items():
            if verbose:
                print("\tProfile {}".format(identifier))

            df = stage.transform(df)

            prof = self._profile_stage(df)
            prof["partial_time"] = prof["total_time"] - temp_total_time
            self._stage_profiles[identifier] = prof

            temp_total_time = prof["total_time"]

    def _profile_with_caching(self, verbose: bool=None) -> None:
        """Make profile with caching enabled for each stage. The input
        dataframe will not be cached.

        Parameters
        ----------
        verbose: bool, optional
            Enable verbose information about current state.

        """

        if verbose:
            print("Profile with caching:")

        df = self._stage_results["input_dataframe"]

        for identifier, stage in self._stage_mapping.items():
            if verbose:
                print("\tProfile {}".format(identifier))

            df = stage.transform(df)
            df.cache()

            prof = self._profile_stage(df)
            cache_time = prof["total_time"]
            self._stage_profiles[identifier]["cached_time"] = cache_time

    def _restore_caching(self) -> None:
        """Restore original caching behaviour. Check each stage's `IsCached`
        parameter and set cache property accordingly.

        """

        for identifier, df in self._stage_results.items():
            if identifier == "input_dataframe":
                continue

            if self._is_cached(self[identifier]):
                df.cache()

    def _profile_report(self) -> None:
        """Collects all profiling information and prints profile report.

        """

        lines = [PROF_L, PROF_H, PROF_L]

        enumeration = enumerate(self._stage_profiles.items())
        for idx, (identifier, prof) in enumeration:
            # format times
            total_time = fmt_time(prof["total_time"])
            partial_time = fmt_time(prof["partial_time"])
            cached_time = fmt_time(prof["cached_time"])

            # shape string
            shape = "{rows} x {cols}".format(rows=prof['rows'],
                                             cols=prof['cols'])

            # line/body string
            line = PROF_B.format(idx=idx,
                                 name=truncate(identifier, 16),
                                 total_time=total_time,
                                 partial_time=partial_time,
                                 cached_time=cached_time,
                                 stage=prof["stage"],
                                 shape=shape)

            lines.append(line)

        lines.append(PROF_L)
        print(*lines, sep="\n")

    def __getitem__(self, value: Union[str, int]) -> TYPE_STAGE:
        """Get stage by index location or label access.

        Index location requires integer value. Label access requires string
        value.

        Parameters
        ----------
        value: str, int
            Integer for index location or string for label access of stages.

        Returns
        -------
        stage: pyspark.ml.Transformer

        """

        if isinstance(value, int):
            return self.stages[value]
        elif isinstance(value, str):
            identifier = self._identify_stage(value)
            return self._stage_mapping[identifier]
        else:
            raise ValueError(ERR_TYPE_ACCESS.format(type(value)))

    def __call__(self, value: Union[str, int]) -> DataFrame:
        """Get stage's dataframe by index location or label access.

        Index location requires integer value. Label access requires string
        value.

        Parameters
        ----------
        value: str, int
            Integer for index location or string for label access of stages.

        Returns
        -------
        df: pyspark.sql.DataFrame
            The dataframe representation of the stage.

        """

        self._is_transformed()

        if isinstance(value, int):
            stage_results = self._stage_results.values()
            return list(stage_results)[value]
        elif isinstance(value, str):
            identifier = self._identify_stage(value)
            return self._stage_results[identifier]
        else:
            raise ValueError(ERR_TYPE_ACCESS.format(type(value)))

    def _is_transformed(self) -> None:
        """Check if pipeline was already run. If not, raise error.

        """

        if not self._stage_results:
            raise ValueError(
                "Pipeline needs to run first via `transform` or parameter "
                "`df` needs to be supplied.")

    @staticmethod
    def _is_cached(stage: TYPE_STAGE) -> bool:
        """Check if given stage has caching enabled or not via `getIsCached`.

        Parameters
        ----------
        stage: pyspark.ml.Transformer
            The stage to check caching for.

        Returns
        -------
        is_cached: bool

        """

        try:
            return stage.getIsCached()
        except AttributeError:
            return False

    def _identify_stage(self, identifier: str) -> str:
        """Identify stage by given identifier. Identifier does not need to be
        a exact match. It will catch all stage identifiers which start with
        given identifier. If more than one stage matches, raise error because
        of ambiguity.

        Parameters
        ----------
        identifier: str
            Identifier to match against all stage identifiers.

        Returns
        -------
        stage_identifier: str
            Full identifier for selected stage.

        """

        stages = [x for x in self._stage_mapping.keys()
                  if x.startswith(identifier)]

        if not stages:
            raise ValueError(
                "Stage with identifier `{identifier}` not found. "
                "Possible identifiers are {options}."
                .format(identifier=identifier,
                        options=self._stage_mapping.keys()))

        if len(stages) > 1:
            raise ValueError(
                "Identifier is ambiguous. More than one stage identified: {}"
                .format(stages))

        return stages[0]

    @staticmethod
    def _get_execution_stage(df: DataFrame) -> str:
        """Extract execution plan stage from `explain` string. Accesses private
        member of pyspark dataframe and may break in future releases. However,
        as of yet, there is no other way to access the execution plan stage.

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Pyspark dataframe for which the current execution plan stage will
            be extracted.

        Returns
        -------
        stage: str

        ToDo: caching causes stages to restart from 0 which needs be evaluated

        """

        explain = df._jdf.queryExecution().simpleString()
        match = REGEX_STAGE.search(explain)
        if match:
            return match.groups()[0]
        else:
            return ""

    def _profile_stage(self, df: DataFrame) -> Dict[str, Union[str, float]]:
        """Profiles dataframe while calling `count` action and return execution
        time, execution plan stage and number of rows and columns.

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Pyspark dataframe to be profiled.

        Returns
        -------
        profile: dict
            Profile results containing `total_time`, `rows`, `cols` and
            `stage`.

        """

        # number of columns
        cols = len(df.columns)

        # total time and count
        ts_start = pd.Timestamp.now()
        rows = df.count()
        ts_end = pd.Timestamp.now()
        total_time = (ts_end - ts_start).total_seconds()

        # execution plan stage
        stage = self._get_execution_stage(df)

        return {"total_time": total_time,
                "rows": rows,
                "cols": cols,
                "stage": stage}

    @staticmethod
    def _create_stage_identifier(stage: TYPE_STAGE) -> str:
        """Given different types of stages, create a unique identifier for
        each stage. Valid pyspark `Transformer` have an uid. Other objects
        will use class name and id.

        Parameters
        ----------
        stage: pyspark.ml.Transformer
            A stage for which a uniqe identifier is returned.

        Returns
        -------
        identifier: str

        """

        try:
            return stage.uid
        except AttributeError:
            if inspect.isclass(stage):
                return "{}_{}".format(stage.__name__, id(stage))
            else:
                return "{}_{}".format(stage.__class__.__name__, id(stage))

    @staticmethod
    def _check_convert_transformer(stage: Any) -> TYPE_STAGE:
        """Ensure given stage is suitable for pipeline usage while checking
        for `transform` attribute. If not and stage is a function, convert
        into `Transformer` instance.

        Parameters
        ----------
        stage: Any
            Any object viable to serve as a transformer.

        Returns
        -------
        converted: pyspark.ml.Transformer
            Object with a `transform` method.

        ToDo: Add conversion for pywrangler.pyspark instances
        ToDo: Allow only Transformer, Wrangler, functions

        """

        if hasattr(stage, "transform"):
            if callable(stage.transform):
                return stage
            else:
                raise ValueError(
                    "Transform method of stage {} is not callable."
                    .format(stage))

        elif inspect.isfunction(stage):
            return func_to_spark_transformer(stage)

        else:
            raise ValueError(
                "Stage '{}' needs to implement `transform` method or "
                "has to be a function.".format(stage))
