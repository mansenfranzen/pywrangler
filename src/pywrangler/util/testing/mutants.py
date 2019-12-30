"""This module contains the data mutants and mutation classes.

"""
from typing import Any, List


class Mutation:
    """Resembles a single mutation of a dataframe.

    """

    def __init__(self, column: str, row: int, value: Any):
        self.column = column
        self.row = row
        self.value = value


class BaseMutant:
    """Base class for all mutants. A mutant produces one or more mutations.

    """

    @property
    def mutations(self) -> List[Mutation]:
        """Returns all mutations produced by a mutant. Needs to be implemented
        by every Mutant.

        """

        raise NotImplementedError


class ValueMutant(BaseMutant):

    def __init__(self, column: str, row: int, value):
        self.column = column
        self.row = row
        self.value = value

    @property
    def mutations(self) -> List[Mutation]:
        mutation = Mutation(column=self.column, row=self.row, value=self.value)
        return [mutation]