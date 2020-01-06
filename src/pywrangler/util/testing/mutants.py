"""This module contains the data mutants and mutation classes.

"""
import itertools
from datetime import datetime
from collections import Counter, defaultdict
from typing import Any, List, NamedTuple, Sequence, Callable, Tuple, Iterable
import random
from string import ascii_letters

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

        return df.modify(modifications)

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
    """Represents a Mutant with a single mutation.

    """

    def __init__(self, column: str, row: int, value):
        self.column = column
        self.row = row
        self.value = value

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:
        mutation = Mutation(column=self.column, row=self.row, value=self.value)
        return [mutation]


class FunctionMutant(BaseMutant):
    """Represents a Mutant which wraps a function that essentially generates
    mutations.

    """

    def __init__(self, func: Callable):
        self.func = func

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:
        return self.func(df)


class RandomMutant(BaseMutant):
    """Creates random mutations with naive values for supported dtypes of
    PlainFrame.

    """

    def __init__(self, count: int = 1, columns: Sequence[str] = None,
                 rows: Sequence[int] = None, seed: int = 1):
        self.count = count
        self.columns = columns
        self.rows = rows
        self.seed = seed

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:

        # set random seed
        random.seed(self.seed)

        # validate columns and rows
        columns = self.columns or df.columns
        rows = self.rows or range(df.n_rows)
        valid_rows = range(df.n_rows)
        invalid_rows = set(rows).difference(valid_rows)
        if invalid_rows:
            raise ValueError("RandomMutant: Invalid rows provided: {}. "
                             "Valid rows are: {}"
                             .format(invalid_rows, valid_rows))

        # validate max count of mutations
        max_count = len(columns) * len(rows)
        count = self.count if self.count <= max_count else max_count

        # generate candidates and draw sample
        candidates = list(itertools.product(columns, rows))
        sample = random.sample(candidates, count)

        return [self.generate_mutation(df, column, row)
                for column, row in sample]

    def generate_mutation(self, df: PlainFrame, column: str,
                          row: int) -> Mutation:
        """Generates mutation from given PlainFrame and single candidate. A
        candidate is specified as a tuple of column name and row index.

        """

        plaincolumn = df.get_column(column)
        value = plaincolumn.values[row]
        new_value = self._random_value(plaincolumn.dtype, value)

        return Mutation(column=column, row=row, value=new_value)

    def _random_value(self, dtype: str, value: Any) -> Any:
        """Helper function to generate a random value.

        """

        def _bool():
            return random.choice([True, False])

        def _int():
            return random.randint(-10, 10)

        def _float():
            return random.random()

        def _str():
            return random.choice(list(ascii_letters))

        def _datetime():
            year = random.randint(datetime.min.year, datetime.max.year)
            return datetime(year=year, month=1, day=1)

        func = {"bool": _bool,
                "int": _int,
                "float": _float,
                "str": _str,
                "datetime": _datetime}[dtype]

        candidate = func()
        while candidate == value:
            candidate = func

        return candidate


class MutantCollection(BaseMutant):
    """Represents a collection of multiple Mutant instances.

    """

    def __init__(self, mutants: Iterable):
        self.mutants = mutants

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:
        mutations = [mutant.generate_mutations(df) for mutant in self.mutants]

        return list(itertools.chain.from_iterable(mutations))
