"""
Handles mapping between color representations and their english
names. For example we want to populate dialogs
with pretty names for matplotlib colors of the current color cyle
but we need to translate between their names and matplotlib
representations.

A slight complication exists for supporting the syntax for the basic
colors in matplotib < v2. Translating to hex colors doesn't yield
the same value as the equivalent CSS name in all cases, for example

basic 'c' = rgb(0, 0.75, 0.75) => #00bfbf
yet 'cyan' is defined in css colors as #00ffff

In this case we accept that a 1->many mapping if names
to hex values and both #00bfbf and #00ffff will return
the string cyan
"""
from __future__ import (absolute_import, division)

from matplotlib import rcParams

from six import iteritems
try:
    from matplotlib.colors import to_hex
except ImportError:
    from matplotlib.colors import colorConverter, rgb2hex

    def to_hex(color):
        return rgb2hex(colorConverter.to_rgb(color))
try:
    from matplotlib.colors import get_named_colors_mapping as mpl_named_colors
except ImportError:
    from matplotlib.colors import cnames

    def mpl_named_colors():
        return cnames

_BASIC_COLORS_HEX_MAPPING = {'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c', 'red': '#d62728',
                             'purple': '#9467bd', 'brown': '#8c564b', 'pink': '#e377c2', 'gray': '#7f7f7f',
                             'olive': '#bcbd22', 'cyan': '#17becf'}


def pretty_name(name):
    """
    Sanitize the color name, removing any prefixes
    :param name: The mpl color name
    :return: A sanitized version for display
    """
    try:
        colon_idx = name.index(":")
        return name[colon_idx+1:]
    except ValueError:
        return name


def named_cycle_colors():
    """
    Retrieve a named list of colors for the current color cycle
    :return: A list of colors as human-readable strings
    """
    axes_prop_cycler = rcParams['axes.prop_cycle']
    try:
        keys = axes_prop_cycler.by_key()
    except AttributeError:
        # cycler < 1 doesn't have by_key but _transpose is the same
        # and depending on a private attribute is okay here as
        # it is only for older versions that won't change
        keys = axes_prop_cycler._transpose()
    return [color_to_name(to_hex(color)) for color in keys['color']]


def name_to_color(name):
    """
    Translate between our string names and the mpl color
    representation
    :param name: One of our known string names
    :return: The string identifier we have chosen
    :raises: ValueError if the color is not known
    """
    try:
        return _BASIC_COLORS_HEX_MAPPING[name]
    except KeyError:
        try:
            return mpl_named_colors()[name]
        except KeyError:
            raise ValueError("Color name {} unknown".format(name))


def color_to_name(color):
    """
    Translate between a matplotlib color representation
    and our string names.
    :param color: Any matplotlib color representation
    :return: The string identifier we have chosen
    :raises: ValueError if the color is not known
    """
    color_as_hex = to_hex(color)
    for name, hexvalue in iteritems(_BASIC_COLORS_HEX_MAPPING):
        if color_as_hex == hexvalue:
            return name
    else:
        for name, value in iteritems(mpl_named_colors()):
            if color_as_hex == to_hex(value):
                return pretty_name(name)
        else:
            raise ValueError("matplotlib color {} unknown".format(color))
