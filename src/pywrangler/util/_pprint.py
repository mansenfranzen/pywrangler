"""This module contains helper functions for printing.

"""

import typing

ITERABLE = typing.Union[typing.List[str], typing.Tuple[str]]
ENUM = typing.Union[ITERABLE, dict]


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
    """Create header with underline.

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
    header: str

    """

    _indent = " " * indent

    _header = _indent + name
    _underline = _indent + underline*len(name) + "\n"

    return _join([_header, _underline])


def enumeration(values: ENUM, indent: int = 0, bullet_char: str = "-") -> str:
    """Create enumeration with bullet points.

    Parameters
    ----------
    values: list, tuple, dict
        Iterable vales. If dict, creates key/value pairs..
    indent: int, optional
        Indentation count.
    bullet_char: str, optional
        Bullet character.

    Returns
    -------
    enumeration: str

    """

    if isinstance(values, dict):
        _values = ["{key}: {value}".format(key=key, value=value)
                   for key, value in sorted(values.items())]
    else:
        _values = values

    with_bullets = ["{} {}".format(bullet_char, x) for x in _values]
    indented = _indent(with_bullets, indent)

    return _join(indented)
