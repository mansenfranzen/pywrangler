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


def test_pretty_file_size():
    pfs = _pprint.pretty_file_size

    assert pfs(1024, precision=1, width=4) == ' 1.0 KiB'
    assert pfs(1024, precision=1, width=4, align="<") == '1.0  KiB'
    assert pfs(1024, precision=1) == '1.0 KiB'
    assert pfs(1024 ** 2, precision=1, width=0) == '1.0 MiB'
    assert pfs(1024 ** 8, precision=2, width=0) == '1.00 YiB'


def test_pretty_time_duration():
    ptd = _pprint.pretty_time_duration

    assert ptd(1.1) == "1.1 s"
    assert ptd(1.59, width=5) == "  1.6 s"
    assert ptd(1.55, width=7, precision=2) == "   1.55 s"
    assert ptd(1.55, width=7, precision=2, align="<") == "1.55    s"
    assert ptd(120, precision=2) == "2.00 min"
    assert ptd(5400, precision=1) == "1.5 h"
    assert ptd(0.5, precision=1) == "500.0 ms"
    assert ptd(0.0005, precision=1) == "500.0 µs"
    assert ptd(0.0000005, precision=1) == "500.0 ns"
