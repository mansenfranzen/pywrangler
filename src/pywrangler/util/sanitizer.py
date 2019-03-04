"""This module contains common helper functions for sanity checks and
conversions.

"""

import collections
from typing import Any, Tuple


def ensure_tuple(values: Any) -> Tuple[Any]:
    """For convenience, some parameters may accept a single value (string
    for a column name) or multiple values (list of strings for column
    names). This function ensures that the output is always a tuple of values.

    Parameters
    ----------
    values: Any
        Input values to be converted to tuples.

    Returns
    -------
    tupled: Tuple[Any]

    """

    # if not iterable, return tuple with single value
    if not isinstance(values, collections.Iterable):
        return (values, )

    # handle single string which is iterable but still is only one value
    elif isinstance(values, str):
        return (values, )

    # anything else should ok to be converted to tuple
    else:
        return tuple(values)
