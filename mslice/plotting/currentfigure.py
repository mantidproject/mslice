"""
This module is responsible for handling which figures are current (will receive next plot operation) and which are
kept (will not be modified until it is made current once again).

The FigureManager in its responsibilities highly resembles the class maptplotlib._pylab_helpers.Gcf, However it
adds the functionality of having multiple 'categories' with each category having it own current window.
This is achieved through use of the supplied decorator 'activate_category' This decorator accepts one parameter  (a
string ) specifying which category a function belongs two. For instance to apply the the the decorator
activate_category('<category>') to the the function pyplot.pcolor would signal that the function pcolor would only apply
to  plots of the category '<category>'. All of this is done through manipulating the the return value of `pyplot.gcf`. gcf
in pyplot returns FigureManager.get_active_figure()

If gcf is called from inside a categorized (decorated) function then it will return the current figure for that
functions category. If there is no current figure for that category then it will create a new figure and return it.

If a new figure is created in a call of gcf that is categorized then the new figure will automatically be assigned
to the category. And it will be set as the current figure for that category.

If gcf is called from inside an uncategorized function then it should return the `active figure` which is defined as
the last plot window to receive any plotting command (regardless of category). This makes sense for functions which
can apply to any plots such as `xlabel`.

If a new figure is created by an uncategorized function then will be 'uncategorized'. If the current 'active figure'
is an uncategorized figure and categorized function is called then it should be returned and then that figure should
be added to the category of the command

Currently there are only two categories ('1d' and '2d') hard coded into the manager.
"""
from __future__ import (absolute_import, division, print_function)

from mslice.plotting.plot_window.plot_figure import PlotFigureManager


class CurrentFigure(object):
    """This is singleton static class to manage the current _figures
    """
    # if there is a current figure it should be both current and active
    _active_category = None
    _category_current_figures = {"1d": None, "2d": None}  # Current _figures recieve decorated commands
    _figures_by_category = {"1d": [], "2d": []}
    _unclassified_figures = []
    _active_figure = None  # Will receive all commands that have a matching decorator or are undecorated
    _figures = {}

    def __init__(self, *args, **kwargs):
        raise Exception("This is a static class singleton. Do not Instantiate it")

    @staticmethod
    def _new_figure(fig_num=None):
        if fig_num is None:
            fig_num = 1
            while any([fig_num == existing_fig_num for existing_fig_num in CurrentFigure._figures.keys()]):
                fig_num += 1
        new_fig = PlotFigureManager(fig_num, CurrentFigure)
        CurrentFigure._figures[fig_num] = new_fig
        CurrentFigure._active_figure = fig_num
        CurrentFigure._unclassified_figures.append(fig_num)
        return CurrentFigure._figures[fig_num], fig_num

    @staticmethod
    def get_figure_number(fig_num=None, create_if_not_found=True):
        CurrentFigure._active_figure = fig_num
        try:
            # make the figure the current figure of its category
            category = CurrentFigure.get_category(fig_num)
            CurrentFigure._category_current_figures[category] = fig_num
        except KeyError:
            # the figure is still uncategorised, Do nothing
            pass

        figure = CurrentFigure._figures.get(fig_num, None)
        if figure or not create_if_not_found:
            return figure
        else:
            # return the figure discard the number
            return CurrentFigure._new_figure(fig_num)[0]

    @staticmethod
    def get_active_figure():
        if CurrentFigure._active_category:
            if CurrentFigure._active_figure in CurrentFigure._unclassified_figures:
                CurrentFigure.assign_figure_to_category(CurrentFigure._active_figure, CurrentFigure._active_category,
                                                        make_current=True)
            elif CurrentFigure._category_current_figures[CurrentFigure._active_category] is None:
                _, num = CurrentFigure._new_figure()
                CurrentFigure.assign_figure_to_category(num, CurrentFigure._active_category, make_current=True)
                CurrentFigure._active_figure = num
            else:
                CurrentFigure._active_figure = CurrentFigure._category_current_figures[CurrentFigure._active_category]
        else:
            if CurrentFigure._active_figure is None:
                fig, num = CurrentFigure._new_figure()
                CurrentFigure._active_figure = num

        return CurrentFigure._figures[CurrentFigure._active_figure]

    @staticmethod
    def _activate_category(category):
        """Sets the active category to the supplied argument, do not call this function directly. Instead use supplied
        Decorator below 'activate_category' """
        CurrentFigure._active_category = category

    @staticmethod
    def _deactivate_category():
        """ Unsets the active category. do not call this function directly. Instead use supplied
        Decorator below 'activate_category' """
        CurrentFigure._active_category = None

    @staticmethod
    def assign_figure_to_category(fig_num, category, make_current=False):
        if fig_num not in CurrentFigure._figures:
            raise ValueError("Figure does not exist")

        if fig_num in CurrentFigure._unclassified_figures:
            CurrentFigure._unclassified_figures.remove(fig_num)

        for a_category in CurrentFigure._figures_by_category:
            if fig_num in CurrentFigure._figures_by_category[a_category]:
                CurrentFigure._figures_by_category[a_category].remove(fig_num)
            if CurrentFigure._category_current_figures == fig_num:
                CurrentFigure._category_current_figures = None

        CurrentFigure._figures_by_category[category].append(fig_num)
        if make_current:
            CurrentFigure._category_current_figures[category] = fig_num
        CurrentFigure.broadcast()

    @staticmethod
    def figure_closed(figure_number):
        """Figure is closed, remove all references to it from all internal list

        If it was the category current or global active figure then set that to None"""
        if CurrentFigure._active_figure == figure_number:
            CurrentFigure._active_figure = None
        for a_category in CurrentFigure._figures_by_category:
            if figure_number in CurrentFigure._figures_by_category[a_category]:
                CurrentFigure._figures_by_category[a_category].remove(figure_number)

            if CurrentFigure._category_current_figures[a_category] == figure_number:
                CurrentFigure._category_current_figures[a_category] = None
        try:
            del CurrentFigure._figures[figure_number]
        except KeyError:
            raise KeyError('The key "%s" does not exist. The figure cannot be closed' % figure_number)

    @staticmethod
    def get_category(figure_number):
        """Return the category of the figure"""
        for category,fig_list in list(CurrentFigure._figures_by_category.items()):
            if figure_number in fig_list:
                figure_category = category
                break
        else:
            raise KeyError("Figure no. %i was not found in any category "%figure_number if figure_number else 0)
            # in-line if handles the case figure_number is None
        return figure_category

    @staticmethod
    def set_figure_as_kept(figure_number):
        # kept figures are just lying around, not really managed much, until they report in as current again
        if CurrentFigure._active_figure == figure_number:
            CurrentFigure._active_figure = None
        try:
            figure_category = CurrentFigure.get_category(figure_number)
        except KeyError:
            figure_category = None

        if figure_category:
            if CurrentFigure._category_current_figures[figure_category] == figure_number:
                CurrentFigure._category_current_figures[figure_category] = None

        CurrentFigure.broadcast(figure_category)

    @staticmethod
    def set_figure_as_current(figure_number):
        try:
            figure_category = CurrentFigure.get_category(figure_number)
        except KeyError:
            figure_category = None
        if figure_category:
            CurrentFigure._category_current_figures[figure_category] = figure_number
        CurrentFigure._active_figure = figure_number
        CurrentFigure.broadcast(figure_category)

    @staticmethod
    def broadcast(category=None):
        """This method will broadcast to all figures in 'category' to update the displayed kept/current status"""
        if category is None:
            broadcast_list = CurrentFigure._figures_by_category
        else:
            broadcast_list = [category]

        for category in broadcast_list:
            for figure_number in CurrentFigure._figures_by_category[category]:

                if CurrentFigure._category_current_figures[category] == figure_number:
                    CurrentFigure._figures[figure_number].set_as_current()

                else:
                    CurrentFigure._figures[figure_number].set_as_kept()

        for figure in CurrentFigure._unclassified_figures:
            if figure == CurrentFigure._active_figure:
                CurrentFigure._figures[figure].set_as_current()
            else:
                CurrentFigure._figures[figure].set_as_kept()

    @staticmethod
    def all_figure_numbers():
        """An iterator over all figure numbers"""
        return list(CurrentFigure._figures.keys())

    @staticmethod
    def all_figures_numbers_in_category(category):
        """Return an iterator over all _figures numbers in a category"""
        return iter(CurrentFigure._figures_by_category[category])

    @staticmethod
    def unclassified_figures():
        """Return an iterator over all unclassified _figures"""
        return iter(CurrentFigure._unclassified_figures)

    @staticmethod
    def reset():
        """Reset all class variables to initial state. This Function exists for testing purposes """
        CurrentFigure._active_category = None
        CurrentFigure._category_current_figures = {"1d": None, "2d": None}  # Current _figures are overplotted
        CurrentFigure._figures_by_category = {"1d": [], "2d": []}
        CurrentFigure._unclassified_figures = []
        CurrentFigure._active_figure = None
        CurrentFigure._figures = {}

    @staticmethod
    def all_figures():
        """Return an iterator over all figures"""
        return list(CurrentFigure._figures.values())

    @staticmethod
    def number_of_figure(figure):
        for key,value in list(CurrentFigure._figures.items()):
            if value == figure:
                return key
        raise ValueError('Figure %s was not recognised'%figure)


