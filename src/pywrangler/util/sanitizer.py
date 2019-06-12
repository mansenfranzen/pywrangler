"""This module contains common helper functions for sanity checks and
conversions.

"""

import collections
from typing import Any, Tuple

import pandas as pd


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

    # None remains None
    if values is None:
        return None

    # if not iterable, return tuple with single value
    elif not isinstance(values, collections.Iterable):
        return (values, )

    # handle exception which are iterable but still count as one value
    elif isinstance(values, (str, pd.DataFrame)):
        return (values, )

    # anything else should ok to be converted to tuple
    else:
        return tuple(values)
