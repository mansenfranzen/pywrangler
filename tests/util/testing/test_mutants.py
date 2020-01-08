"""This module contains data mutant and mutations tests.

"""
from datetime import datetime

import pytest
from pywrangler.util.testing import mutants, plainframe


def test_mutation():
    """Test correct mutant instantiation.

    """

    mutation = mutants.Mutation("foo", 1, "bar")

    assert mutation.column == "foo"
    assert mutation.row == 1
    assert mutation.value == "bar"
    assert mutation.key == ("foo", 1)


def test_base_mutant():
    """Test BaseMutant functionality.

    """

    # check duplicated mutations
    m1 = mutants.Mutation("foo", 1, "bar")
    m2 = mutants.Mutation("foo", 1, "far")
    m3 = mutants.Mutation("bar", 0, "foo")
    m4 = mutants.Mutation("foo", 0, "bar")

    with pytest.raises(ValueError):
        mutants.BaseMutant._check_duplicated_mutations([m1, m2])

    mutants.BaseMutant._check_duplicated_mutations([m1, m3])

    # check invalid mutations
    df = plainframe.PlainFrame.from_dict({"foo:str": ["bar"]})

    with pytest.raises(ValueError):
        mutants.BaseMutant._check_valid_mutations([m1], df)  # check row

    with pytest.raises(ValueError):
        mutants.BaseMutant._check_valid_mutations([m3], df)  # check column

    mutants.BaseMutant._check_valid_mutations([m4], df)


def test_value_mutant():
    """Test ValueMutant functionality.

    """

    df = plainframe.PlainFrame.from_dict({"foo:str": ["bar"]})
    df_test = plainframe.PlainFrame.from_dict({"foo:str": ["foo"]})
    mutant = mutants.ValueMutant("foo", 0, "foo")
    mutation = mutants.Mutation("foo", 0, "foo")

    assert mutant.generate_mutations(df) == [mutation]
    assert mutant.mutate(df) == df_test


def test_function_mutant():
    """Test FunctionMutant functionality.

    """

    df = plainframe.PlainFrame.from_dict({"foo:str": ["bar"]})
    df_test = plainframe.PlainFrame.from_dict({"foo:str": ["foo"]})
    mutation = mutants.Mutation("foo", 0, "foo")

    def custom_func(df):
        return [mutation]

    mutant = mutants.FunctionMutant(custom_func)

    assert mutant.generate_mutations(df) == [mutation]
    assert mutant.mutate(df) == df_test


def test_random_mutant():
    """Test RandomMutant functionalty.

    """

    df = plainframe.PlainFrame.from_dict({"foo:str": ["bar"],
                                          "bar:int": [1]})

    # test invalid column
    mutant = mutants.RandomMutant(columns=["not_exists"])
    with pytest.raises(ValueError):
        mutant.mutate(df)

    # test column
    mutant = mutants.RandomMutant(columns=["foo"])
    df_mutated = mutant.mutate(df)
    assert df_mutated.get_column("foo").values[0] != "bar"
    assert df_mutated.get_column("bar").values[0] == 1

    # test invalid row
    mutant = mutants.RandomMutant(rows=[2])
    with pytest.raises(ValueError):
        mutant.mutate(df)

    # test row
    df_rows = plainframe.PlainFrame.from_dict({"foo:str": ["bar", "foo"]})
    mutant = mutants.RandomMutant(rows=[1])
    df_mutated = mutant.mutate(df_rows)
    assert df_mutated.get_column("foo").values[0] == "bar"
    assert df_mutated.get_column("foo").values[0] != "foo"

    # test count
    mutant = mutants.RandomMutant(count=1)
    df_mutated = mutant.mutate(df)
    assert ((df_mutated.get_column("foo").values[0] != "bar") !=
            (df_mutated.get_column("bar").values[0] != 1))

    # test max count
    mutant = mutants.RandomMutant(count=100)
    df_mutated = mutant.mutate(df)
    assert df_mutated.get_column("foo").values[0] != "bar"
    assert df_mutated.get_column("bar").values[0] != 1

    # test random funcs for all types
    random = mutants.RandomMutant._random_value
    date = datetime(2019, 1, 1)
    assert isinstance(random("bool", True), bool)
    assert isinstance(random("int", 1), int)
    assert isinstance(random("float", 1.1), float)
    assert isinstance(random("str", "foo"), str)
    assert isinstance(random("datetime", date), datetime)

    assert random("bool", True) is not True
    assert random("int", 1) != 1
    assert random("float", 1.1) != 1.1
    assert random("str", "foo") != "foo"
    assert random("datetime", date) != date



def test_collection_mutant():
    """Test MutantCollection functionality.

    """

    # test combination
    df = plainframe.PlainFrame.from_dict({"foo:str": ["foo", "foo"]})
    value_mutant = mutants.ValueMutant("foo", 0, "bar")
    func = lambda _: [mutants.Mutation("foo", 1, "bar")]
    func_mutant = mutants.FunctionMutant(func)

    result = [mutants.Mutation("foo", 0, "bar"),
              mutants.Mutation("foo", 1, "bar")]

    df_result = plainframe.PlainFrame.from_dict({"foo:str": ["bar", "bar"]})

    mutant_collection = mutants.MutantCollection([value_mutant, func_mutant])
    assert mutant_collection.generate_mutations(df) == result
    assert mutant_collection.mutate(df) == df_result


def test_mutant_assertions():
    """Test invalid type changes due to mutations.

    """

    df = plainframe.PlainFrame.from_dict({"foo:str": ["foo", "foo"]})

    mutant = mutants.ValueMutant("foo", 1, 2)
    with pytest.raises(TypeError):
        mutant.mutate(df)
