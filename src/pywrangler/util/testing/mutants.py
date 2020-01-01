"""This module contains the data mutants and mutation classes.

"""
from collections import Counter, defaultdict
from typing import Any, List, NamedTuple, Sequence

from pywrangler.util.testing.plainframe import PlainFrame

ImmutableMutation = NamedTuple("ImmutableMutation", [("column", str),
                                                     ("row", int),
                                                     ("value", Any)])


class Mutation(ImmutableMutation):
    """Resembles a single mutation of a dataframe which essentially represents
    a data modification of a single cell of a dataframe. Hence, a mutation is
    fully specified via three values: a column, a row and a new value.

    The column is always given via label (string). The row is always given via
    an index (integer) because plainframe does not have labeled indices. The
    new value may be of any type.

    """

    @property
    def key(self):
        return self.column, self.row


class BaseMutant:
    """Base class for all mutants. A mutant produces one or more mutations.

    """

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:
        """Returns all mutations produced by a mutant given a plainframe. Needs
        to be implemented by every Mutant.

        """

        raise NotImplementedError

    def mutate(self, df: PlainFrame) -> PlainFrame:
        """Modifies given plainframe with inherent mutations.

        """

        mutations = self.generate_mutations(df)
        self._check_duplicated_mutations(mutations)
        self._check_valid_mutations(mutations, df)

        modifications = defaultdict(dict)
        for mutation in mutations:
            modifications[mutation.column][mutation.row] = mutation.value

        return df.modifiy(modifications)

    @staticmethod
    def _check_duplicated_mutations(mutations: Sequence[Mutation]):
        """Validate unique mutations to prevent overwriting data modifications.

        Raises ValueError.

        """

        keys = [mutation.key for mutation in mutations]
        counter = Counter(keys)

        duplicated = [key for key, count in counter.items() if count > 1]
        if duplicated:
            raise ValueError("Duplicated mutations found: following "
                             "mutations have identical column/row "
                             "specifications which causes unpredictable "
                             "modifications: {}"
                             .format(duplicated))

    @staticmethod
    def _check_valid_mutations(mutations: Sequence[Mutation], df: PlainFrame):
        """Validate that mutations are applicable to plainframe.

        """

        def has_column(column: str) -> bool:
            return column in df.columns

        def has_row(row: int) -> bool:
            return row <= df.n_rows

        for mutation in mutations:
            if not has_column(mutation.column):
                raise ValueError("Mutation ({}) is not applicable to given "
                                 "PlainFrame. Column '{}' does not exist."
                                 .format(mutation, mutation.column))

            if not has_row(mutation.row):
                raise ValueError("Mutation ({}) is not applicable to given "
                                 "PlainFrame. Row '{}' does not exist."
                                 .format(mutation, mutation.row))


class ValueMutant(BaseMutant):

    def __init__(self, column: str, row: int, value):
        self.column = column
        self.row = row
        self.value = value

    @property
    def mutations(self) -> List[Mutation]:
        mutation = Mutation(column=self.column, row=self.row, value=self.value)
        return [mutation]


class FunctionMutant(BaseMutant):
    pass


class RandomMutant(BaseMutant):
    def __init__(self, count=1, columns=None, rows=None):
        pass
