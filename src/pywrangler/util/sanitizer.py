"""This module contains common helper functions for sanity checks and
conversions.

"""

import collections
from typing import Any, List, Tuple, Type, Union

import pandas as pd

ITER_TYPE = Union[List[Any], Tuple[Any], None]


def ensure_iterable(values: Any, seq_type: Type = list) -> ITER_TYPE:
    """For convenience, some parameters may accept a single value (string
    for a column name) or multiple values (list of strings for column
    names). Other functions always require a list or tuple of strings. Hence,
    this function ensures that the output is always an iterable of given
    `constructor` type (list or tuple) while taking care of exceptions just
    like strings.

    Parameters
    ----------
    values: Any
        Input values to be converted to tuples.
    seq_type: {list, tuple}
        Iterable class to return.

    Returns
    -------
    iterable: List[Any], Tuple[Any]

    """

    # None remains None
    if values is None:
        return None

    # if not iterable, return iterable with single value
    elif not isinstance(values, collections.Iterable):
        return seq_type([values])

    # handle exception which are iterable but still count as one value
    elif isinstance(values, (str, pd.DataFrame)):
        return seq_type([values])

    # anything else should ok to be converted to tuple/list
    else:
        return seq_type(values)
