"""This module contains functionality to check optional and mandatory imports.
It aims to provide useful error messages if optional dependencies are missing.
"""

import importlib
import sys
from functools import wraps
from typing import Callable


def raise_if_missing(import_name):
    """Checks for available import and raises with more detailed error
    message if not given.

    Parameters
    ----------
    import_name: str

    """
    try:
        importlib.import_module(import_name)

    except ImportError as e:
        msg = ("The requested functionality requires '{dep}'. "
               "However, '{dep}' is not available in the current "
               "environment with the following interpreter: "
               "'{interpreter}'. Please install '{dep}' first.\n\n"
               .format(dep=import_name, interpreter=sys.executable))

        raise type(e)(msg) from e


def requires(*deps: str) -> Callable:
    """Decorator for callables to ensure that required dependencies are met.
    Provides more useful error message if dependency is missing.

    Parameters
    ----------
    deps: list
        List of dependencies to check.

    Returns
    -------
    decorated: callable

    Examples
    --------

    >>> @requires("dep1", "dep2")
    >>> def func(a):
    >>>     return a

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for dep in deps:
                raise_if_missing(dep)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_available(*deps: str) -> bool:
    """Check if given dependencies are available.

    Parameters
    ----------
    deps: list
        List of dependencies to check.

    Returns
    -------
    available: bool

    """

    for dep in deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            return False

    return True
