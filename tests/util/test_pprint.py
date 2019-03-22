"""Test printing helpers.

"""

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


def test_enumeration_dict_align_values_false():
    test_input = {"a": 1, "bb": 2}
    test_output = '- a: 1\n- bb: 2'

    assert _pprint.enumeration(test_input, align_values=False) == test_output


def test_enumeration_dict_align_values():
    test_input = {"a": 1, "bb": 2}
    test_output = '-  a: 1\n- bb: 2'

    assert _pprint.enumeration(test_input) == test_output


def test_enumeration_dict_align_values_with_align_width():
    test_input = {"a": 1, "bb": 2}
    test_output = '-   a: 1\n-  bb: 2'

    assert _pprint.enumeration(test_input, align_width=3) == test_output


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


def test_sizeof():
    assert _pprint.sizeof(1024, precision=1, width=0) == '1.0 KiB'
    assert _pprint.sizeof(1024, precision=1) == '   1.0 KiB'
    assert _pprint.sizeof(1024, precision=1, align="<") == '1.0    KiB'
    assert _pprint.sizeof(1024 ** 2, precision=1, width=0) == '1.0 MiB'
    assert _pprint.sizeof(1024 ** 8, precision=2, width=0) == '1.00 YiB'
