"""This module contains type definitions.

"""

from typing import Iterable, Union, Optional

T_STR_OPT_MUL = Optional[Iterable[str]]
T_STR_OPT_SING_MUL = Optional[Union[str, Iterable[str]]]

TYPE_COLUMNS = T_STR_OPT_SING_MUL
TYPE_ASCENDING = Union[bool, Iterable[bool]]
