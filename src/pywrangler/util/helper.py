"""This module contains commonly used helper functions or classes.

"""

from typing import Callable


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
