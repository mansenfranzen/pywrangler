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
    row index starts with 0. The new value may be of any type.

    """

    @property
    def key(self):
        return self.column, self.row


class BaseMutant:
    """Base class for all mutants. A mutant produces one or more mutations.

    """

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:
        """Returns all mutations produced by a mutant given a PlainFrame. Needs
        to be implemented by every Mutant. This is essentially the core of
        every mutant.

        Parameters
        ----------
        df: PlainFrame
            PlainFrame to generate mutations from.

        Returns
        -------
        mutations: list
            List of Mutation instances.

        """

        raise NotImplementedError

    def mutate(self, df: PlainFrame) -> PlainFrame:
        """Modifies given PlainFrame with inherent mutations and returns new,
        modifed PlainFrame.

        Parameters
        ----------
        df: PlainFrame
            PlainFrame to be modified.

        Returns
        -------
        modified: PlainFrame

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
            return row <= df.n_rows-1

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

    Attributes
    ----------
    column: str
        Name of the column.
    row: int
        Index of the row.
    value: Any
        The new value to be used.

    """

    def __init__(self, column: str, row: int, value: Any):
        self.column = column
        self.row = row
        self.value = value

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:
        """Returns a single mutation.

        Parameters
        ----------
        df: PlainFrame
            PlainFrame to generate mutations from.

        Returns
        -------
        mutations: list
            List of Mutation instances.

        """

        mutation = Mutation(column=self.column, row=self.row, value=self.value)

        return [mutation]


class FunctionMutant(BaseMutant):
    """Represents a Mutant which wraps a function that essentially generates
    mutations.

    Attributes
    ----------
    func: callable
        A function to be used as a mutation generation method.

    """

    def __init__(self, func: Callable):
        self.func = func

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:
        """Delegates the mutation generation to a custom function to allow
        all possible mutation generation.

        Parameters
        ----------
        df: PlainFrame
            PlainFrame to generate mutations from.

        Returns
        -------
        mutations: list
            List of Mutation instances.

        """

        return self.func(df)


class RandomMutant(BaseMutant):
    """Creates random mutations with naive values for supported dtypes of
    PlainFrame. Randomness is controlled via an explicit seed to allow
    reproducibility. Mutation generation may be narrowed to given rows or
    columns. The number of distinct mutations may also be specified.

    Attributes
    ----------
    count: int, optional
        The number of mutations to be executed.
    columns: sequence, optional
        Restrict mutations to provided columns, if given.
    rows: sequence, optional
        Restrict mutations to provided rows, if given.
    seed: int, optional
        Set the seed for the random generator.

    """

    def __init__(self, count: int = 1, columns: Sequence[str] = None,
                 rows: Sequence[int] = None, seed: int = 1):
        self.count = count
        self.columns = columns
        self.rows = rows
        self.seed = seed

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:
        """Generates population of all possible mutations and draws a sample of
        it.

        Parameters
        ----------
        df: PlainFrame
            PlainFrame to generate mutations from.

        Returns
        -------
        mutations: list
            List of Mutation instances.

        """

        # set random seed
        random.seed(self.seed)

        # validate columns and rows
        columns = self._get_validated_columns(df)
        rows = self._get_validated_rows(df)

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
        """Generates single mutation from given PlainFrame for a given
        candidate. A candidate is specified via column name and row index.

        Parameters
        ----------
        df: PlainFrame
            PlainFrame to generate mutations from.
        column: str
            Identifies relevant column of mutation.
        row: int
            Identifies relevant row of mutation.

        Returns
        -------
        mutation: Mutation

        """

        plaincolumn = df.get_column(column)
        value = plaincolumn.values[row]
        new_value = self._random_value(plaincolumn.dtype, value)

        return Mutation(column=column, row=row, value=new_value)

    @staticmethod
    def _random_value(dtype: str, original_value: Any) -> Any:
        """Helper function to generate a random value given original value
        and dtype.

        Parameters
        ----------
        dtype: str
            Defines the dtype of the new value.
        original_value: Any
            Represents original value

        Returns
        -------
        new_value: Any
            Generated new random value.

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
        while candidate == original_value:
            candidate = func()

        return candidate

    def _get_validated_rows(self, df: PlainFrame) -> List[int]:
        """Provide validated rows. Provided rows which are not present in given
        PlainFrame will raise a ValueError.

        Parameters
        ----------
        df: PlainFrame
            PlainFrame to generate mutations from.

        Returns
        -------
        rows: list
            List of validated rows (integers)

        """

        rows = self.rows or list(range(df.n_rows))

        valid_rows = range(df.n_rows)
        invalid_rows = set(rows).difference(valid_rows)
        if invalid_rows:
            raise ValueError("RandomMutant: Invalid rows provided: {}. "
                             "Valid rows are: {}"
                             .format(invalid_rows, valid_rows))

        return rows

    def _get_validated_columns(self, df: PlainFrame) -> List[str]:
        """Provide validated columns. Provided columns which are not present in
        given PlainFrame will raise a ValueError.

        Parameters
        ----------
        df: PlainFrame
            PlainFrame to generate mutations from.

        Returns
        -------
        rows: list
            List of validated columns (strings).

        """

        columns = self.columns or df.columns
        invalid_columns = set(columns).difference(df.columns)
        if invalid_columns:
            raise ValueError("RandomMutant: Invalid columns provided: {}. "
                             "Valid columns are: {}"
                             .format(invalid_columns, df.columns))

        return columns


class MutantCollection(BaseMutant):
    """Represents a collection of multiple Mutant instances.

    Attributes
    ----------
    mutants: sequence
        List of mutants.

    """

    def __init__(self, mutants: Sequence):
        self.mutants = mutants

    def generate_mutations(self, df: PlainFrame) -> List[Mutation]:
        """Collects all mutations generated by included Mutants.

        Parameters
        ----------
        df: PlainFrame
            PlainFrame to generate mutations from.

        Returns
        -------
        mutations: list
            List of Mutation instances.

        """

        mutations = [mutant.generate_mutations(df) for mutant in self.mutants]

        return list(itertools.chain.from_iterable(mutations))
