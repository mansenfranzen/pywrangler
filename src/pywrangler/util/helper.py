"""This module contains commonly used helper functions or classes.

"""

import inspect
from typing import Callable, List

from pywrangler.util.types import T_STR_OPT_MUL


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
