"""This module contains type definitions.

"""

from typing import Iterable, Union

T_STR_OPT_MUL = Union[Iterable[str], None]
T_STR_OPT_SING_MUL = Union[str, Iterable[str], None]

TYPE_COLUMNS = T_STR_OPT_SING_MUL
TYPE_ASCENDING = Union[bool, Iterable[bool], None]
