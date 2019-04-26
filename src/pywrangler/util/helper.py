"""This module contains commonly used helper functions or classes.

"""

import inspect
from typing import Callable, List

from pywrangler.util.types import T_STR_OPT_MUL


def cached_property(method: Callable) -> property:
    """Decorated method will be called only on first access to calculate a
    cached property value. After that, the cached value is returned.

    Parameters
    ---------
    method: Callable
        Getter method to be lazily evaluated.

    Returns
    -------
    property

    Notes
    -----
    Credit goes to python-pptx: https://github.com/scanny/python-pptx/blob/master/pptx/util.py

    """  # noqa: E501

    cache_attr_name = '__{}'.format(method.__name__)
    docstring = method.__doc__

    def get_prop_value(obj):
        try:
            return getattr(obj, cache_attr_name)
        except AttributeError:
            value = method(obj)
            setattr(obj, cache_attr_name, value)
            return value

    return property(get_prop_value, doc=docstring)


def get_param_names(func: Callable,
                    ignore: T_STR_OPT_MUL = None) -> List[str]:
    """Retrieve all parameter names for given function.

    Parameters
    ----------
    func: Callable
        Function for which parameter names should be retrieved.
    ignore: iterable, None, optional
        Parameter names to be ignored. For example, `self` for `__init__`
        functions.

    Returns
    -------
    param_names: list
        List of parameter names.

    """

    ignore = ignore or []

    signature = inspect.signature(func)
    parameters = signature.parameters.values()

    param_names = [x.name for x in parameters if x.name not in ignore]

    return param_names
