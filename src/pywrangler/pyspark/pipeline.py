"""This module adds extended pipeline functionality for pyspark.

"""

import copy
import inspect
import re
from collections.abc import KeysView
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union
)

import numpy as np
import pandas as pd
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.param.shared import Param, Params
from pyspark.sql import DataFrame

from pywrangler.pyspark.base import PySparkWrangler
from pywrangler.util.sanitizer import ensure_iterable

TYPE_STAGE = Union[PySparkWrangler, Transformer, Callable]
TYPE_IDENTIFIER = Union[int, str, Transformer, slice]
TYPE_PARAM_DICT = Dict[str, Union[Callable, Param]]

StageProperties = NamedTuple("StageProperties", [("name", str),
                                                 ("cols", int),
                                                 ("stage_count", int),
                                                 ("cached", bool),
                                                 ("uid", str)])

StageProfile = NamedTuple("StageProfile", [("idx", str),
                                           ("name", str),
                                           ("total_time", float),
                                           ("rows", int),
                                           ("cols", int),
                                           ("stage_count", int),
                                           ("cached", bool),
                                           ("uid", str)])

StageDescription = NamedTuple("StageDescription", [("idx", str),
                                                   ("name", str),
                                                   ("doc", str),
                                                   ("cols", int),
                                                   ("stage_count", int),
                                                   ("cached", bool),
                                                   ("uid", str)])

ERR_TYPE_ACCESS = "Value has incorrect type '{}' (integer or string allowed)."


class StageTransformerConverter:
    """Wrap arbitrary pipeline stage object and allow conversion to a valid
    pyspark `Transformer` to comply with pyspark's pipeline/transformer API.

    Accepts instances of type `Transformer`, `Wrangler` and native python
    functions.

    """

    def __init__(self, stage: TYPE_STAGE):
        """Create reference of stage object.

        Parameters
        ----------
        stage: Any
            Arbitrary pipeline stage object.

        """

        self.stage = stage

    def convert(self) -> Transformer:
        """Ensure stage is suitable for pipeline usage while allowing
        only instances of type `Transformer`, `Wrangler` or native python
        functions. Objects which are not of type `Transformer` will be
        converted into it.

        Returns
        -------
        converted: pyspark.ml.Transformer
            Object with a `transform` method.

        """

        if isinstance(self.stage, Transformer):
            return self.stage
        elif isinstance(self.stage, PySparkWrangler):
            return self.convert_wrangler()
        elif inspect.isfunction(self.stage):
            return self.convert_function()

        else:
            raise ValueError(
                "Stage needs to be a `Transformer`, `PySparkWrangler` "
                "or a native python function. However, '{}' was given."
                .format(type(self.stage)))

    def convert_wrangler(self) -> Transformer:
        """Convert a `PySparkWrangler` into a pyspark `Transformer`.

        Creates a deep copy of the original wrangler instance to leave it
        unchanged. The original API is lost and the pyspark `Transformer` API
        applies.

        Temporarily creates a new sublcass of type `Transformer` during
        runtime while ensuring that all keyword arguments of the wrangler
        are mapped to corresponding `Param` identifiers with required getter
        and setter methods for the resulting `Transformer` class.

        Returns
        -------
        transformer: pyspark.ml.Transformer

        """

        def _transform(self, df):
            self._wrangler.set_params(**self.getParams())
            return self._wrangler.transform(df)

        cls_name = self.stage.__class__.__name__
        cls_dict = {"_wrangler": copy.deepcopy(self.stage),
                    "_transform": _transform,
                    "__doc__": self.stage.__doc__}

        # get parameters
        params = self.stage.get_params()
        params_dict = self._create_param_dict(params.keys())
        cls_dict.update(params_dict)

        return self._instantiate_transformer(cls_name, cls_dict, params)

    def convert_function(self) -> Transformer:
        """Convert a native python function into a pyspark `Transformer`
        instance. Expects the first parameter to be positional representing the
        input dataframe.

        Temporarily creates a new sublcass of type `Transformer` during
        runtime while ensuring that all keyword arguments of the input
        function are mapped to corresponding `Param` identifiers with
        required getter and setter methods for the resulting `Transformer`
        class.

        Returns
        -------
        transformer: pyspark.ml.Transformer

        """

        def _transform(self, df):
            return self._func(df, **self.getParams())

        cls_name = self.stage.__name__
        cls_dict = {"_func": staticmethod(self.stage),
                    "_transform": _transform,
                    "__doc__": self.stage.__doc__}

        # get parameters
        signature = inspect.signature(self.stage)
        params = signature.parameters.values()
        params = {x.name: x.default for x in params
                  if not x.default == inspect._empty}

        params_dict = self._create_param_dict(params.keys())
        cls_dict.update(params_dict)

        return self._instantiate_transformer(cls_name, cls_dict, params)

    def _create_param_dict(self, parameters: KeysView) -> TYPE_PARAM_DICT:
        """Create getter/setter methods and Param attributes for given
        parameters to comply `pyspark.ml.Transformer` API.

        Parameters
        ----------
        parameters: KeysView
            Contains the names of the parameters.

        Returns
        -------
        param_dict: Dict[str, Union[Callable, Param]]
            Dictionary containing the parameter setter/getter and Param
            attributes.

        """

        def setParams(self, **kwargs):
            return self._set(**kwargs)

        def getParams(self):
            params = self.extractParamMap().items()
            kwargs = {key.name: value for key, value in params}
            return kwargs

        param_dict = {"setParams": setParams,
                      "getParams": getParams}

        # create setter/getter and Param instances
        for parameter in parameters:
            param_dict.update(self._create_getter_setter(parameter))
            param_dict[parameter] = Param(Params._dummy(), parameter, "")

        return param_dict

    @staticmethod
    def _instantiate_transformer(name: str,
                                 dicts: Dict[str, Any],
                                 params: Dict[str, Any]) -> Transformer:
        """Create subclass of `pyspark.ml.Transformer` during runtime with name
        `name` and methods/attributes `dicts`. Create instance of it and
        configure it with given parameters `params`.

        Parameters
        ----------
        name: str
            Name of the class.
        dicts: Dict[str, Any]
            All methods/attributes of the class.
        params: Dict[str, Any]
            All parameters to be set for a new instance of this class.

        Returns
        -------
        transformer_instance: Transformer

        """

        transformer_class = type(name, (Transformer,), dicts)
        transformer_instance = transformer_class()
        transformer_instance._set(**params)

        return transformer_instance

    @staticmethod
    def _create_getter_setter(name: str) -> Dict[str, Callable]:
        """Helper function to create getter and setter methods for parameters
        of `Transformer` class for given parameter name.

        Parameters
        ----------
        name: str
            The name of the parameter.

        Returns
        -------
        param_dict: Dict[str, Callable]
            Dictionary containing the getter/setter methods for single
            parameter.

        """

        def setter(self, value):
            """Using the `self._set` is the default implementation for setting
            user-supplied params for `Transformer`

            """

            return self._set(**{name: value})

        def getter(self):
            """Passing the `Param` value of the parameter to `getOrDefault` is
            the default implementation of `Transformer`.

            """

            return self.getOrDefault(getattr(self, name))

        return {"get{name}".format(name=name): getter,
                "set{name}".format(name=name): setter}


class PipelineLocator:
    """Composite for `Pipeline` that manages position and label based access of
    pipeline stages and corresponding dataframe transformations.

    Position based access is equivalent to index location lookup. Label based
    access is equivalent to identifier lookup.

    """

    def __init__(self, pipeline: 'Pipeline'):
        """Pipeline locator keeps track of stage's identifier-idx mapping via
        `self.identifiers`.

        Parameters
        ----------
        pipeline: Pipeline
            Parent pipeline object to be composite of.

        """

        self.pipeline = pipeline

        enumerated = enumerate(self.pipeline.stages)
        self.identifiers = {stage.uid: idx for idx, stage in enumerated}

    def map_identifier_to_index(self, identifier: str) -> int:
        """Find corresponding index location for given identifier. Identifier
        does not need to be an exact match. Case insensitive, partial match
        suffices.

        Parameters
        ----------
        identifier: str
            Identifier string to match against stage identifiers.

        Returns
        -------
        idx: int
            Index location of matched identifier.

        """

        validated = self.search_validate_identifier(identifier)

        return self.identifiers[validated]

    def search_validate_identifier(self, identifier: str) -> str:
        """Search stages by given `identifier`. Matching is case
        insensitive and substring equality suffices.

        If more than one stage matches, raise error because of
        ambiguity.

        If no stage matches, raises error.

        Parameters
        ----------
        identifier: str
            Identifier to match against all stage identifiers.

        Returns
        -------
        stage_identifier: str
            Full identifier for selected stage.

        """

        stages = [x for x in self.identifiers.keys()
                  if identifier.lower() in x.lower()]

        if not stages:
            raise ValueError(
                "Stage with identifier `{identifier}` not found. "
                "Possible identifiers are {options}."
                .format(identifier=identifier,
                        options=self.identifiers.keys()))

        if len(stages) > 1:
            raise ValueError(
                "Identifier is ambiguous. More than one stage identified: {}"
                .format(stages))

        return stages[0]

    def get_index_location(self, value: TYPE_IDENTIFIER) -> int:
        """Return validated stage index location.

        Parameters
        ----------
        value: int, str, Transformer
            If string is provided, checks for matching stage identifier. If
            integer is provided, checks out of range error. If Transformer is
            provided, checks if transformer instance is part of pipeline.

        Returns
        -------
        idx: int
            Index location of selected stage.

        """

        if isinstance(value, str):
            return self.map_identifier_to_index(value)

        elif isinstance(value, int):
            stage_cnt = len(self.identifiers)
            if value > stage_cnt:
                raise IndexError(
                    "Pipeline has only {} stages.".format(stage_cnt))
            return value

        elif isinstance(value, Transformer):
            if value not in self.pipeline.stages:
                raise ValueError("Stage '{}' is not part of pipeline"
                                 .format(value))
            return self.pipeline.stages.index(value)

        else:
            raise ValueError(ERR_TYPE_ACCESS.format(type(value)))

    def get_stage(self, value: TYPE_IDENTIFIER) -> Transformer:
        """Return pipeline stage for given index or stage identifier.

        Parameters
        ----------
        value: int, str, Transformer
            Identifies stage via index location or identifier substring.

        Returns
        -------
        stage: pyspark.ml.Transformer

        """

        idx = self.get_index_location(value)
        return self.pipeline.stages[idx]

    def get_transformation(self, value: TYPE_IDENTIFIER) -> DataFrame:
        """Return pipeline stage's transformation for given index or
        stage identifier.

        Parameters
        ----------
        value: int, str
            Identifies stage via index location or identifier substring.

        Returns
        -------
        stage: pyspark.sql.DataFrame

        """

        transformer = self.pipeline._transformer

        if not transformer:
            raise ValueError("Dataframe representation of selected stage is "
                             "not available yet. Please execute pipeline "
                             "first via `transform`.")

        idx = self.get_index_location(value)
        return transformer.transformations[idx]


class PipelineCacher:
    """Composite for `Pipeline` that handles stage caching on pipeline level.
    Stores and modifies cache properties of stages.

    Pipeline caches are applied on the result of a stage transformation. This
    is different from caching on stage level. Stage level caching is applied
    within each stage and not necessarily on the result of the stage.

    Pipeline caching has no influence on stage caching except if the stage
    caches the very last result it returns. In this case, caches are
    interchangeably.

    """

    def __init__(self, pipeline: 'Pipeline'):
        """Pipeline cacher keeps track of stages for which caching is enabled
        via `self._store`.

        Parameters
        ----------
        pipeline: Pipeline
            Parent pipeline object to be composite of.

        """

        self.pipeline = pipeline
        self._store = set()

    def enable(self, stages: List[TYPE_IDENTIFIER]) -> None:
        """Enable pipeline caching for given stages. Stage can be identified
        via index, identifier or stage itself.

        If pipeline was already transformed, enables caching on existing
        dataframe representations. However, `transform` has to be called again
        for the execution plan of the pipeline's result dataframe to respect
        caching changes.

        Parameters
        ----------
        stages: iterable
            Iterable of int, str or Transformer.

        """

        stages = ensure_iterable(stages)

        for stage in stages:
            idx = self.pipeline._loc.get_index_location(stage)
            self._store.add(idx)

            if self.pipeline._transformer:
                self.pipeline(idx).cache()

    def disable(self, stages: List[TYPE_IDENTIFIER]) -> None:
        """Disable pipeline caching for given stages. Stage can be identified
        via index, identifier or stage itself.

        If pipeline was already transformed, disables caching on existing
        dataframe representations. However, `transform` has to be called again
        for the execution plan of the pipeline's result dataframe to respect
        caching changes.

        Parameters
        ----------
        stages: iterable
            Iterable of int, str or Transformer.

        """

        stages = ensure_iterable(stages)

        for stage in stages:
            idx = self.pipeline._loc.get_index_location(stage)

            try:
                self._store.remove(idx)
            except KeyError:
                raise ValueError("'{}' does not exist in cache and hence"
                                 "cannot be disabled.".format(stage))

            if self.pipeline._transformer:
                self.pipeline(idx).unpersist(blocking=True)

    def clear(self) -> None:
        """Remove all stage caches on pipeline level.

        If pipeline was already transformed, unpersists all already cached
        stage's dataframes.

        """

        if self.pipeline._transformer:
            for idx in self._store:
                self.pipeline(idx).unpersist(blocking=True)

        self._store.clear()

    @property
    def enabled(self) -> List[Transformer]:
        """Return all stages with caching enabled on pipeline level
        in correct order.

        """

        return [self.pipeline.stages[idx]
                for idx in sorted(self._store)]


class PipelineTransformer:
    """Composite for `Pipeline` that manages the actual dataframe
    transformation performed by all stages in sequence for given input
    dataframe while incorporating pipeline caching.

    """

    def __init__(self, pipeline: 'Pipeline'):
        """Pipeline transformer keeps track of all stages dataframe
        transformations via `self.transformation`. In addition, the input
        dataframe is referenced via `self.input_df`.

        Parameters
        ----------
        pipeline: Pipeline
            Parent pipeline object to be composite of.

        """

        self.pipeline = pipeline
        self.transformations = []
        self.input_df = None

    def transform(self, df: DataFrame) -> DataFrame:
        """Performs dataframe transformation for given input dataframe while
        respecting pipeline caches.

        Previous transformations are overridden and hence will be lost.

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Input dataframe to apply transformations to.

        Returns
        -------
        df_result: pyspark.sql.DataFrame
            Resulting dataframe after all transformations have been applied.

        """

        self.transformations.clear()
        self.input_df = df

        cache_enabled = self.pipeline.cache.enabled

        for stage in self.pipeline.stages:
            df = stage.transform(df)

            if stage in cache_enabled:
                df.cache()

            self.transformations.append(df)

        return df

    def __iter__(self) -> Iterator:
        """Allow transformer to be iterable. Simply returns an iterator of
        `transformations`.

        """

        return iter(self.transformations)

    def __bool__(self) -> bool:
        """Return if transformation was already run.

        """

        return len(self.transformations) > 0


class PipelineProfiler:
    """Represents a profile of a given pipeline. Executes each stage in order
    and collects information about execution time, execution plan stage, shape
    of the resulting dataframe and caching.

    """

    regex_stage = re.compile(r"\*\((\d+)\) ")

    def __init__(self, pipeline: 'Pipeline'):
        """Keeps track of all profiles stages via `self.profiles`.

        Parameters
        ----------
        pipeline: Pipeline
            Pipeline object to be profiled.

        """

        self.pipeline = pipeline

    def profile(self, df: Optional[DataFrame] = None) -> pd.DataFrame:
        """Profiles each pipeline stage and provides information about
        execution time, execution plan stage and stage dataframe shape.

        Parameters
        ----------
        df: pyspark.sql.DataFrame, optional
            If provided, profiles pipeline on given dataframe. If not given,
            uses already existing pipeline transformer object.

        Returns
        -------
        profile: pd.DataFrame

        """

        return self._execute("profile", df)

    def describe(self, df: Optional[DataFrame] = None) -> pd.DataFrame:
        """Describes each pipeline stage and provides information about
        execution plan stage, number of columns and docs.

        Parameters
        ----------
        df: pyspark.sql.DataFrame, optional
            If provided, profiles pipeline on given dataframe. If not given,
            uses already existing pipeline transformer object.

        Returns
        -------
        description: pd.DataFrame

        """

        return self._execute("describe", df)

    def _execute(self, method: str,
                 df: Optional[DataFrame] = None) -> pd.DataFrame:
        """Generic function to either describe or profile given pipeline.
        Describing the pipeline does not run any actions. In contrast,
        profiling will actually call actions and may take some time.

        Parameters
        ----------
        method: str
            Choose either `describe` or `profile`.
        df: pyspark.sql.DataFrame, optional
            If provided, profiles pipeline on given dataframe. If not given,
            uses already existing pipeline transformer object.

        """

        methods = {
            "describe": self._get_stage_description,
            "profile": self._get_stage_profile
        }

        caller = methods[method]

        # ensure existing input dataframe
        if df is None and not self.pipeline._transformer:
            raise ValueError("Please provide input dataframe via `df` "
                             "or run `transform` method first.")

        if df is not None:
            transformer = PipelineTransformer(self.pipeline)
            transformer.transform(df)
        else:
            transformer = self.pipeline._transformer

        # reset profiler
        results = []

        # initial profile of input dataframe
        start_profile = caller(transformer.input_df)
        results.append(start_profile)

        # subsequent stage profiles
        for idx, df_stage in enumerate(transformer):
            results.append(caller(df_stage, idx))

        results = [result._asdict() for result in results]
        return pd.DataFrame(results)

    def _get_stage_properties(self,
                              df_stage: DataFrame,
                              idx: Optional[int] = None) -> StageProperties:
        """Provides general stage properties like identifier, stage execution
        count, number of columns and caching.

        Parameters
        ----------
        df_stage: pyspark.sql.DataFrame
            Dataframe representation of stage.
        idx: integer, None, optional
            If idx is given, resembles a valid pipeline stage. If not,
            represents input dataframe.

        Returns
        -------
        stage_properties: StageProperties

        """

        if idx is None:
            name = "Input dataframe"
            uid = ""
        else:
            name = self.pipeline.stages[idx].__class__.__name__
            uid = self.pipeline.stages[idx].uid

        execution_stage_count = self._get_execution_stage_count(df_stage)
        cols = len(df_stage.columns)
        cached = df_stage.is_cached

        return StageProperties(name, cols, execution_stage_count, cached, uid)

    def _get_stage_profile(self,
                           df_stage: DataFrame,
                           idx: Optional[int] = None) -> StageProfile:
        """Profile pipeline stage's dataframe and collect index, identifier,
        total time, number of rows and columns, execution plan stage and
        dataframe caching.

        Parameters
        ----------
        df_stage: pyspark.sql.DataFrame
            Dataframe representation of stage.
        idx: integer, None, optional
            If idx is given, resembles a valid pipeline stage. If not,
            represents input dataframe.

        Returns
        -------
        profile: StageProfile

        """

        stage_properties = self._get_stage_properties(df_stage, idx)
        rows, total_time = self._get_rows_and_execution_time(df_stage)

        return StageProfile(str(idx),
                            stage_properties.name,
                            total_time,
                            rows,
                            stage_properties.cols,
                            stage_properties.stage_count,
                            stage_properties.cached,
                            stage_properties.uid)

    def _get_stage_description(self,
                               df_stage: DataFrame,
                               idx: Optional[int] = None) -> StageDescription:
        """Describe pipeline stages and collect index, identifier,
        number of columns, stage execution count, doc string and caching.

        Parameters
        ----------
        df_stage: pyspark.sql.DataFrame
            Dataframe representation of stage.
        idx: integer, None, optional
            If idx is given, resembles a valid pipeline stage. If not,
            represents input dataframe.

        Returns
        -------
        description: StageDescription

        """

        stage_properties = self._get_stage_properties(df_stage, idx)

        if idx is not None:
            doc_string = self.pipeline.stages[idx].__doc__
        else:
            doc_string = ""

        return StageDescription(str(idx),
                                stage_properties.name,
                                doc_string,
                                stage_properties.cols,
                                stage_properties.stage_count,
                                stage_properties.cached,
                                stage_properties.uid)

    def _get_execution_stage_count(self, df: DataFrame) -> int:
        """Extract execution plan stage from `explain` string. All maximum
        execution plan stages are summed up to account for resets due to
        caching.

        Accesses private member of pyspark dataframe and may break in future
        releases. However, as of yet, there is no other way to access the
        execution plan stage.

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Pyspark dataframe for which the current execution plan stage will
            be extracted.

        Returns
        -------
        stage_count: int

        """

        # get explain string
        explain = df._jdf.queryExecution().simpleString()
        stages = self.regex_stage.findall(explain)

        # convert to numpy array
        integers = map(int, stages)
        array = np.array(list(integers))

        # remove all stages with larger successor
        shift = np.hstack((0, array[:-1]))
        decreases = array - shift
        mask = decreases > 0

        result = np.sum(array[mask])

        # sum up all local maxima
        return int(result)

    @staticmethod
    def _get_rows_and_execution_time(df: DataFrame) -> Tuple[int, float]:
        """Profiles dataframe while calling `count` action and return execution
        time, execution plan stage and number of rows.

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

        # total time and count
        ts_start = pd.Timestamp.now()
        rows = df.count()
        ts_end = pd.Timestamp.now()
        total_time = (ts_end - ts_start).total_seconds()

        return rows, total_time


class Pipeline(PipelineModel):
    """Represents an extended subclass of `pyspark.ml.PipelineModel` which adds
    several convenient features for more general ETL purposes. Concretely, it
    allows to use pyspark wranglers and native python functions as transformers
    that resemble a valid data transformation stage. To comply with the
    `pyspark.ml.Transformer` interface, pyspark wrangler and python functions
    are automatically converted into valid `Transformer` instances. Keyword
    arguments of python functions and configuration of pyspark wranglers will
    be available as normal parameters of the resulting pyspark `Transformer`.

    In addition, the `describe` method gives a brief overview of the stages.
    The `profile` method provides timings and shapes of each stage. Also,
    `__call__` is implemented to conveniently access the resulting dataframe
    representation of each stage while `__getitem__` allows to access the
    `Transformer` instance of each stage.

    Each pipeline instance may be provided with an explicit documentation
    string.

    Parameters
    ----------
    stages: iterable
        Contains the stages for the pipleline.
    doc: str, optional
        Provide optional doc string for the pipeline.

    """

    def __init__(self, stages: List, doc: Optional[str] = None):
        """Instantiate pipeline. Validate/convert stage input.

        """

        converted = [StageTransformerConverter(stage).convert()
                     for stage in stages]

        super().__init__(tuple(converted))

        # public
        self.cache = PipelineCacher(self)
        self.doc = doc

        # private
        self._loc = PipelineLocator(self)
        self._transformer = PipelineTransformer(self)

    def profile(self, df: Optional[DataFrame] = None) -> pd.DataFrame:
        """Executes each stage in order and collects information about
        execution time, execution plan stage, shape of the resulting dataframe
        and caching.

        Parameters
        ----------
        df: pyspark.sql.DataFrame, optional
            If provided, profiles pipeline on given dataframe. If not given,
            uses already existing pipeline transformer object.

        Returns
        -------
        profile: pd.DataFrame

        """

        return PipelineProfiler(self).profile(df)

    def describe(self, df: Optional[DataFrame] = None) -> pd.DataFrame:
        """Describes each stage in order and collects information about
        execution plan stage, number of columns and caching.

        Parameters
        ----------
        df: pyspark.sql.DataFrame, optional
            If provided, profiles pipeline on given dataframe. If not given,
            uses already existing pipeline transformer object.

        Returns
        -------
        description: pd.DataFrame

        """

        return PipelineProfiler(self).describe(df)

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

        return self._transformer.transform(df)

    def __getitem__(self, value: TYPE_IDENTIFIER) -> \
            Union[Transformer, 'Pipeline']:
        """Get stage by index location/label access or create a sliced copy of
        the pipeline.

        Index location requires integer value. Label access requires string
        value. Sliced copy of the pipeline requires a slice.

        Parameters
        ----------
        value: str, int, slice
            Integer for index location or string for label access of stages. If
            slice is given, creates a sliced copy of the pipeline.

        Returns
        -------
        stage: pyspark.ml.Transformer

        """

        if isinstance(value, slice):
            start = value.start or 0
            stop = value.stop or len(self.stages)

            idx_start = self._loc.get_index_location(start)
            idx_end = self._loc.get_index_location(stop)

            stages = self.stages[idx_start:idx_end + 1]
            stages = [copy.deepcopy(stage) for stage in stages]

            return Pipeline(stages, self.doc)

        else:
            return self._loc.get_stage(value)

    def __call__(self, value: TYPE_IDENTIFIER) -> DataFrame:
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

        return self._loc.get_transformation(value)

    def __repr__(self):
        tpl = "Pipeline (Stages: {}, Uid: {}, Doc: {})"
        return tpl.format(len(self.stages), self.uid, self.doc)

    def __str__(self):
        return self.__repr__()
