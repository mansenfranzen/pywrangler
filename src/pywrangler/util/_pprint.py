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
        If dict is provided, align all values to the same column. The longest
        key defines the exact position.
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


def sizeof(size: float, precision: int = 2, align: str = ">",
           width=None) -> str:
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

    if width is None:
        width = precision + 5

    kwargs = dict(width=width, precision=precision, align=align)

    # iterate units (multiples of 1024 bytes)
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(size) < 1024.0:
            return template.format(size=size, unit=unit, **kwargs)
        size /= 1024.0

    else:
        return template.format(size=size, unit='Yi', **kwargs)
