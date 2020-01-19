"""This module contains the BaseWrangler definition and the wrangler base
classes including wrangler descriptions and parameters.

"""
import inspect
from abc import ABC, abstractmethod

from pywrangler.util import _pprint
from pywrangler.util.helper import get_param_names


class BaseWrangler(ABC):
    """Defines the basic interface common to all data wranglers.

    In analogy to sklearn transformers (see link below), all wranglers have to
    implement `fit`, `transform` and `fit_transform` methods. In addition,
    parameters (e.g. column names) need to be provided via the `__init__`
    method. Furthermore, `get_params` and `set_params` methods are required for
    grid search and pipeline compatibility.

    The `fit` method contains optional fitting (e.g. compute mean and variance
    for scaling) which sets training data dependent transformation behaviour.
    The `transform` method includes the actual computational transformation.
    The `fit_transform` either applies the former methods in sequence or adds a
    new implementation of both with better performance. The `__init__` method
    should contain any logic behind parameter parsing and conversion.

    In contrast to sklearn, wranglers do only accept dataframes like objects
    (like pandas/pyspark/dask dataframes) as inputs to `fit` and `transform`.
    The relevant columns and their respective meaning is provided via the
    `__init__` method. In addition, wranglers may accept multiple input
    dataframes with different shapes. Also, the number of samples may also
    change between input and output (which is not allowed in sklearn). The
    `preserves_sample_size` indicates whether sample size (number of rows) may
    change during transformation.

    The wrangler's employed computation engine is given via
    `computation_engine`.

    See also
    --------
    https://scikit-learn.org/stable/developers/contributing.html

    """

    @property
    @abstractmethod
    def preserves_sample_size(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def computation_engine(self) -> str:
        raise NotImplementedError

    def get_params(self) -> dict:
        """Retrieve all wrangler parameters set within the __init__ method.

        Returns
        -------
        param_dict: dictionary
            Parameter names as keys and corresponding values as values

        """

        base_classes = [cls for cls in inspect.getmro(self.__class__)
                        if issubclass(cls, BaseWrangler)]

        ignore = ["self", "args", "kwargs"]
        param_names = []
        for cls in base_classes[::-1]:
            param_names.extend(get_param_names(cls.__init__, ignore))

        param_dict = {x: getattr(self, x) for x in param_names}

        return param_dict

    def set_params(self, **params):
        """Set wrangler parameters

        Parameters
        ----------
        params: dict
            Dictionary containing new values to be updated on wrangler. Keys
            have to match parameter names of wrangler.

        Returns
        -------
        self

        """

        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter {} for wrangler {}. '
                                 'Check the list of available parameters '
                                 'with `wrangler.get_params().keys()`.'
                                 .format(key, self))

            setattr(self, key, value)

        return self

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def transform(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):

        template = '{wrangler_name} ({computation_engine})\n\n{parameters}'\

        parameters = (_pprint.header("Parameters", 3) +
                      _pprint.enumeration(self.get_params(), 3))

        _repr = template.format(wrangler_name=self.__class__.__name__,
                                computation_engine=self.computation_engine,
                                parameters=parameters)

        if not self.preserves_sample_size:
            _repr += "\n\n   Note: Does not preserve sample size."

        return _repr
