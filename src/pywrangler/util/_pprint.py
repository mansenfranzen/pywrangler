"""This module contains helper functions for printing.

"""

import re
import textwrap
from typing import Any, List, Tuple, Union

ITERABLE = Union[List[str], Tuple[str]]
ENUM = Union[ITERABLE, dict]

REGEX_REMOVE_WHITESPACES = re.compile(r"\s{2,}")


def _join(lines: ITERABLE) -> str:
    """Join given lines.

    Parameters
    ----------
    lines: list, tuple
        Iterable to join.

    Returns
    -------
    joined: str

    """

    return "\n".join(lines)


def _indent(lines: ITERABLE, indent: int = 3) -> list:
    """Indent given lines and optionally join.

    Parameters
    ----------
    lines: list, tuple
        Iterable to indent.
    indent: int, optional
        Indentation count.

    """

    spacing = " " * indent
    return [spacing + x for x in lines]


def header(name: str, indent: int = 0, underline: str = "-") -> str:
    """Create columns with underline.

    Parameters
    ----------
    name: str
        Name of title.
    indent: int, optional
        Indentation count.
    underline: str, optional
        Underline character.

    Returns
    -------
    columns: str

    """

    _indent = " " * indent

    _header = _indent + name
    _underline = _indent + underline * len(name) + "\n"

    return _join([_header, _underline])


def enumeration(values: ENUM, indent: int = 0, bullet_char: str = "-",
                align_values: bool = True, align_width: int = 0) -> str:
    """Create enumeration with bullet points.

    Parameters
    ----------
    values: list, tuple, dict
        Iterable vales. If dict, creates key/value pairs..
    indent: int, optional
        Indentation count.
    bullet_char: str, optional
        Bullet character.
    align_values: bool, optional
        If dict is provided, align all identifiers to the same column. The
        longest key defines the exact position.
    align_width: int, optional
        If dict is provided and `align_values` is True, manually set the align
        width.

    Returns
    -------
    enumeration: str

    """

    if isinstance(values, dict):
        fstring = "{key:>{align_width}}: {value}"
        if align_values and not align_width:
            align_width = max([len(x) for x in values.keys()])

        _values = [fstring.format(key=key,
                                  value=value,
                                  align_width=align_width)

                   for key, value in sorted(values.items())]
    else:
        _values = values

    with_bullets = ["{} {}".format(bullet_char, x) for x in _values]
    indented = _indent(with_bullets, indent)

    return _join(indented)


def pretty_file_size(size: float, precision: int = 2, align: str = ">",
                     width: int = 0) -> str:
    """Helper function to format size in human readable format.

    Parameters
    ----------
    size: float
        The size in bytes to be converted into human readable format.
    precision: int, optional
        Define shown precision.
    align: {'<', '^', '>'}, optional
        Format align specifier.
    width: int
        Define maximum width for number.

    Returns
    -------
    human_fmt: str
        Human readable representation of given `size`.

    Notes
    -----
    Credit to https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size

    """  # noqa: E501

    template = "{size:{align}{width}.{precision}f} {unit}B"
    kwargs = dict(width=width, precision=precision, align=align)

    # iterate units (multiples of 1024 bytes)
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(size) < 1024.0:
            return template.format(size=size, unit=unit, **kwargs)
        size /= 1024.0

    return template.format(size=size, unit='Yi', **kwargs)


def pretty_time_duration(seconds: float, precision: int = 1, align: str = ">",
                         width: int = 0) -> str:
    """Helper function to format time duration in human readable format.

    Parameters
    ----------
    seconds: float
        The size in seconds to be converted into human readable format.
    precision: int, optional
        Define shown precision.
    align: {'<', '^', '>'}, optional
        Format align specifier.
    width: int
        Define maximum width for number.

    Returns
    -------
    human_fmt: str
        Human readable representation of given `seconds`.

    """

    template = "{time_delta:{align}{width}.{precision}f} {unit}"

    units = [('year', 60 * 60 * 24 * 365),
             ('month', 60 * 60 * 24 * 30),
             ('d', 60 * 60 * 24),
             ('h', 60 * 60),
             ('min', 60),
             ('s', 1),
             ('ms', 1e-3),
             ('µs', 1e-6),
             ('ns', 1e-9)]

    # catch 0 value
    if seconds == 0:
        return template.format(time_delta=0,
                               align=align,
                               width=width,
                               precision=0,
                               unit="s")

    # catch negative value
    if seconds < 0:
        sign = -1
        seconds = abs(seconds)
    else:
        sign = 1

    for unit_name, unit_seconds in units:
        if seconds > unit_seconds:
            time_delta = seconds / unit_seconds
            return template.format(time_delta=sign * time_delta,
                                   align=align,
                                   width=width,
                                   precision=precision,
                                   unit=unit_name)


def textwrap_docstring(dobject: Any, width: int = 70) -> List[str]:
    """Extract doc string from object and textwrap it with given width. Remove
    double whitespaces.

    Parameters
    ----------
    dobject: Any
        Object to extract doc string from.
    width: int, optional
        Length of text values to wrap doc string.

    Returns
    -------
    Wrapped doc string as list of lines.

    """

    if not dobject.__doc__:
        return []

    sanitized = REGEX_REMOVE_WHITESPACES.sub(" ", dobject.__doc__).strip()
    return textwrap.wrap(sanitized, width=width)


def truncate(string: str, width: int, ending: str = "...") -> str:
    """Truncate string to be no longer than provided width. When truncated, add
    add `ending` to shortened string as indication of truncation.

    Parameters
    ----------
    string: str
        String to be truncated.
    width: int
        Maximum amount of characters before truncation.
    ending: str, optional
        Indication string of truncation.

    Returns
    -------
    Truncated string.

    """

    if not len(string) > width:
        return string

    length = width - len(ending)
    return string[:length] + ending
