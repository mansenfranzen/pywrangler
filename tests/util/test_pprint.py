"""Test printing helpers.

"""

import pytest

from pywrangler.util import _pprint


def test_join():

    test_input = ["a", "b", "c"]
    test_output = "a\nb\nc"

    assert _pprint._join(test_input) == test_output


def test_indent():

    test_input = ["a", "b", "c"]
    test_output = ["   a", "   b", "   c"]

    assert _pprint._indent(test_input, 3) == test_output


def test_header():

    test_input = "Header"
    test_output = 'Header\n------\n'

    assert _pprint.header(test_input) == test_output


def test_header_with_indent():

    test_input = "Header"
    test_output = '   Header\n   ------\n'

    assert _pprint.header(test_input, indent=3) == test_output


def test_header_with_underline():

    test_input = "Header"
    test_output = 'Header\n======\n'

    assert _pprint.header(test_input, underline="=") == test_output


def test_enumeration_dict():

    test_input = {"a": 1, "b": 2}
    test_output = '- a: 1\n- b: 2'

    assert _pprint.enumeration(test_input) == test_output


def test_enumeration_list():

    test_input = ["note 1", "note 2"]
    test_output = '- note 1\n- note 2'

    assert _pprint.enumeration(test_input) == test_output


def test_enumeration_list_with_indent():

    test_input = ["note 1", "note 2"]
    test_output = '    - note 1\n    - note 2'

    assert _pprint.enumeration(test_input, indent=4) == test_output


def test_enumeration_list_with_bullet():

    test_input = ["note 1", "note 2"]
    test_output = 'o note 1\no note 2'

    assert _pprint.enumeration(test_input, bullet_char="o") == test_output
