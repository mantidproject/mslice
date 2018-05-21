"""
A collection of decorators that apply to pyplot functions
"""
from __future__ import (absolute_import)
from functools import wraps


def activate_category(cfig, category):
    """Mark a function as part of the given category. For details
    of the category mechanism see the docstring on the currentfigure
    module

    :param cfig: A reference to the object responsible for managing the category status
    :param category: '1d' or '2d' to denote the category of the plot produced
    """
    def activate_impl(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            cfig._activate_category(category)
            return_value = function(*args, **kwargs)
            cfig._deactivate_category()
            return return_value
        return wrapper

    return activate_impl


def draw_colorbar(gcf, colorbar):
    """Add this decorator to the definition of a pyplot function
    to enable a colorbar to be automatically generated

    :param gcf: A reference to a function that will return the current figure when called
    :param colorbar: A reference to the pyplot function that draws a colorbar in to the current axes
    """
    def draw_colorbar_impl(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            function(*args, **kwargs)
            cb = getattr(gcf(),'_colorbar_axes',None)
            if cb:
                colorbar(cax=gcf().get_axes()[1])
            else:
                cb = colorbar()
                gcf()._colorbar_axes = cb
        return wrapper
    return draw_colorbar_impl
