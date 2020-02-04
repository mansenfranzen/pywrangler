"""This module contains pyspark specific types.

"""
from typing import Union, Iterable, Optional

from pyspark.sql import Column

TYPE_PYSPARK_COLUMNS = Optional[
    Union[str, Column, Iterable[str], Iterable[Column]]]
