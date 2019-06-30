"""This module adds extended pipeline functionality for pyspark.

"""

import inspect
import re
import textwrap
from collections import OrderedDict

import pandas as pd
from pyspark.ml import PipelineModel, Transformer
from pyspark.ml.param.shared import Param, Params

from pywrangler.util._pprint import pretty_time_duration


def _create_getter_setter(name):
    """Helper function to create getter and setter methods
    for parameters of `Transformer` class for given parameter
    name.

    """

    def setter(self, value):
        return self._set(**{name: value})

    def getter(self):
        return self.getOrDefault(getattr(self, name))

    return {f"get{name}": getter,
            f"set{name}": setter}


def func_to_spark_transformer(func):
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


REGEX_STAGE = re.compile(r".*?\*\((\d+)\).*")
REGEX_CLEAR = re.compile(r"\s{2,}")


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

    In addtion, the `_transform` method checks for an optionally defined
    `IsCached` parameter to cache intermediate results for each stage. This
    enables the possibility to cache specific stages.
    The `desribe` method gives a brief overview of the stages.
    The `profile` method allows to get detailed timings and provides
    indicators for possible caching benefits for each stage.

    Also, `__call__` is implemented to conveniently access dataframe
    representations for each stage and `__getitem__` allows to access
    stage representation.

    """

    def __init__(self, stages):
        stages = [self._check_convert_transformer(stage)
                  for stage in stages]

        super().__init__(stages)

        self.stage_dict = OrderedDict()
        for stage in self.stages:
            identifier = self._create_stage_identifier(stage)
            self.stage_dict[identifier] = stage

        self.stage_results = OrderedDict()

    def _transform(self, dataset):

        self.stage_results.clear()
        self.stage_results["input_dataframe"] = dataset

        for identifier, stage in self.stage_dict.items():
            dataset = stage.transform(dataset)

            try:
                is_cached = stage.getIsCached()
            except AttributeError:
                is_cached = False

            if is_cached:
                dataset.cache()

            self.stage_results[identifier] = dataset

        return dataset

    def describe(self):

        header = "| ({idx}) - {identifier}, {cols} columns, stage {stage}"
        body = "| {text:76} |"
        line = "+" + "-" * 78 + "+"
        arrow = r"""                                      ||
                                     \||/
                                      \/"""

        for idx, (identifier, stage) in enumerate(self.stage_dict.items()):
            df = self(identifier)
            cols = len(df.columns)
            exec_stage = self._get_execution_stage(df)
            title = header.format(idx=idx,
                                  identifier=identifier,
                                  cols=cols,
                                  stage=exec_stage)
            title = title.ljust(79, " ") + "|"

            if not stage.__doc__:
                docs = []
            else:
                cleared = REGEX_CLEAR.sub(" ", stage.__doc__)
                docs = [body.format(text=wrapped)
                        for wrapped in textwrap.wrap(cleared, width=78)]
                docs.append(line)

            print(line, title, line, *docs, arrow, sep="\n")

    def profile(self):
        self._is_transformed()
        self.stage_profiles = OrderedDict()
        self._profile_without_caching()
        self._profile_with_caching()
        self._restore_caching()
        self._profile_report()

    def _profile_without_caching(self):
        """Make profile without caching enabled.

        """

        df = self.stage_results["input_dataframe"]

        # make sure df is not cached
        df.unpersist(blocking=True)

        # profile initial stage
        prof = self._profile_stage(df)
        prof["partial_time"] = 0
        prof["cached_time"] = 0
        self.stage_profiles["input_dataframe"] = prof

        # keep ascendent time
        temp_total_time = prof["total_time"]

        for identifier, stage in self.stage_dict.items():
            df = stage.transform(df)
            df.unpersist(blocking=True)

            prof = self._profile_stage(df)
            prof["partial_time"] = prof["total_time"] - temp_total_time
            self.stage_profiles[identifier] = prof

            temp_total_time = prof["total_time"]

    def _profile_with_caching(self):
        """Make profile with caching enabled for each stage.

        """

        df = self.stage_results["input_dataframe"]

        df.cache()
        df.count()

        for identifier, stage in self.stage_dict.items():
            df = stage.transform(df)
            df.cache()

            prof = self._profile_stage(df)
            cache_time = prof["total_time"]
            self.stage_profiles[identifier]["cached_time"] = cache_time

    def _restore_caching(self):
        """Unpersist all previously persisted dataframes during profiling
        and restore original caching properties.

        """

        for identifier, dataset in self.stage_results.items():
            if identifier == "input_dataframe":
                dataset.unpersist()
                continue

            stage = self[identifier]

            try:
                is_cached = stage.getIsCached()
            except AttributeError:
                is_cached = False

            if not is_cached:
                dataset.unpersist(blocking=True)

    def _profile_report(self):
        """Prints profile report.

        """

        header = "| Idx |    Identifier    | Total time | Partial time |" \
                 " Cached time | Stage | Shape (rows x cols) |"
        underline = "+-----+------------------+------------+--------------+" \
                    "-------------+-------+---------------------+"
        tpl = "| {idx:^3} | {name:^16} | {total_time:^10} | " \
              "{partial_time:^12} | {cached_time:^11} | {stage:^5} |" \
              " {shape:^19} |"

        print(underline)
        print(header)
        print(underline)

        for idx, (identifier, prof) in enumerate(self.stage_profiles.items()):

            if len(identifier) > 16:
                identifier = identifier[:13] + "..."

            total_time = pretty_time_duration(prof["total_time"], align=">")
            partial_time = pretty_time_duration(prof["partial_time"],
                                                align=">")
            cached_time = pretty_time_duration(prof["cached_time"], align=">")
            shape = f"{prof['rows']} x {prof['cols']}"

            print(tpl.format(idx=idx,
                             name=identifier,
                             total_time=total_time,
                             partial_time=partial_time,
                             cached_time=cached_time,
                             stage=prof["stage"],
                             shape=shape))

        print(underline)

    def __getitem__(self, value):
        """Convenient stage access method via location (given integer) or
        label (given string).

        """

        if isinstance(value, int):
            return self.stages[value]
        elif isinstance(value, str):
            identifier = self._identify_stage(value)
            return self.stage_dict[identifier]
        else:
            raise ValueError(f"Value has incorrect type '{type(value)}'. "
                             f"Allowed types are integer and string.")

    def __call__(self, value):
        """Convenient stage's dataframe representation access method via
        location (given integer) or label (given string).

        """

        self._is_transformed()

        if isinstance(value, int):
            return list(self.stage_results.values())[value]
        elif isinstance(value, str):
            identifier = self._identify_stage(value)
            return self.stage_results[identifier]
        else:
            raise ValueError(f"Value has incorrect type '{type(value)}'. "
                             f"Allowed types are integer and string.")

    def _is_transformed(self):
        """Check if pipeline was already run. If not, raise error.

        """

        if not self.stage_results:
            raise ValueError(
                "Pipeline needs to run first via `transform` or parameter "
                "`df` needs to be supplied.")

    def _identify_stage(self, identifier):
        """Identify stage by given identifier. Identifier does not need to be
        a full match. It will catch all stage identifiers which start with
        given label. If more than one stage identifier matches, raise
        error.

        """

        stages = [x for x in self.stage_dict.keys()
                  if x.startswith(identifier)]

        if not stages:
            raise ValueError(
                f"Stage with identifier `{identifier}` not found. Possible "
                f"identifiers are {self.stage_dict.keys()}")

        if len(stages) > 1:
            raise ValueError(
                f"Identifier is ambiguous. More than one stage "
                f"identified: {stages}")

        return stages[0]

    @staticmethod
    def _get_execution_stage(df):
        """Extract execution plan stage from `explain` string.

        """

        explain = df._jdf.queryExecution().simpleString()
        match = REGEX_STAGE.search(explain)
        if match:
            return match.groups()[0]
        else:
            return ""

    def _profile_stage(self, df):
        """Apply `count` on given `df` and return execution time, number of
        rows and columns.

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
    def _create_stage_identifier(stage):
        """Given different types of stages, create a unique identifier for
        each stage. Valid pyspark `Transformer` have an uid. Other objects
        will use class name and id.

        """

        try:
            return stage.uid
        except AttributeError:
            if inspect.isclass(stage):
                return f"{stage.__name__}_{id(stage)}"
            else:
                return f"{stage.__class__.__name__}_{id(stage)}"

    @staticmethod
    def _check_convert_transformer(stage):
        """Ensure given stage is suitable for pipeline usage while checking
        for `transform` attribute. If not and stage is a function, convert
        into `Transformer` instance.

        """

        if hasattr(stage, "transform"):
            if callable(stage.transform):
                return stage
            else:
                raise ValueError(
                    f"Transform method of stage {stage} is not callable.")

        elif inspect.isfunction(stage):
            return func_to_spark_transformer(stage)

        else:
            raise ValueError(
                f"Stage '{stage}' needs to implement `transform` method or "
                f"has to be a function.")
