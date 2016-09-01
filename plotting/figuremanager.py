from __builtin__ import staticmethod

from plotting import get_figure_class
PlotFigureManager = get_figure_class( )
NO_FIGURE = -1723


class FigureManager(object):
    """This is singleton static class to manage the current _figures

    Do not instantiate this class"""
    # if there is a current figure it should be both current and active
    _active_category = None
    _category_active_figures = {"1d": NO_FIGURE, "2d": NO_FIGURE}  # active _figures recieve commands
    _category_current_figures = {"1d": NO_FIGURE, "2d": NO_FIGURE}  # Current _figures are overplotted
    _figures_by_category = {"1d": [], "2d": []}
    _unclassified_figures = []
    _active_figure = NO_FIGURE
    _figures = {}

    def __init__(self, *args, **kwargs):
        raise Exception("This is a static class singleton. Do not Instantiate it")

    @staticmethod
    def _new_figure(fig_num=None):
        if fig_num is None:
            fig_num = 1
            while any([fig_num == existing_fig_num for existing_fig_num in FigureManager._figures.keys()]):
                fig_num += 1
        new_fig = PlotFigureManager( fig_num, FigureManager )
        FigureManager._figures[fig_num] = new_fig
        FigureManager._active_figure = fig_num
        FigureManager._unclassified_figures.append(fig_num)
        return FigureManager._figures[fig_num], fig_num

    @staticmethod
    def get_figure_number(fig_num=None,create_if_not_found=True):
        FigureManager._active_figure = fig_num
        try:
            category = FigureManager.get_category(fig_num)
            FigureManager._category_active_figures[category] = fig_num
        except KeyError:
            # the figure is still uncategorised
            pass

        figure = FigureManager._figures.get(fig_num, None)
        if figure or not create_if_not_found:
            return figure
        else:
            # return the figure discard the number
            return FigureManager._new_figure(fig_num)[0]

    @staticmethod
    def get_active_figure():
        if FigureManager._active_figure in FigureManager._unclassified_figures:
            return FigureManager._figures[FigureManager._active_figure]
        if FigureManager._active_category:
            if FigureManager._category_active_figures[FigureManager._active_category] == NO_FIGURE:
                fig,num = FigureManager._new_figure()
                FigureManager.assign_figure_to_category(num,FigureManager._active_category)
                FigureManager._category_active_figures[FigureManager._active_category] = num
                FigureManager._active_figure = num
            else:
                FigureManager._active_figure = FigureManager._category_active_figures[FigureManager._active_category]
        if FigureManager._active_figure == NO_FIGURE:
            fig, num = FigureManager._new_figure()
            FigureManager._active_figure = num
        return FigureManager._figures[FigureManager._active_figure]

    @staticmethod
    def _activate_category(category):
        """Sets the active category to the supplied argument, do not call this function directly. Instead use supplied
        Decorator below 'activate_category' """
        FigureManager._active_category = category

    @staticmethod
    def _deactivate_category():
        """ Unsets the active category. do not call this function directly. Instead use supplied
        Decorator below 'activate_category' """
        FigureManager._active_category = None

    @staticmethod
    def assign_figure_to_category(fig_num,category,make_current=False):
        if fig_num not in FigureManager._figures.keys():
            raise ValueError("Figure does not exist")
        if fig_num in FigureManager._unclassified_figures:
            FigureManager._unclassified_figures.remove(fig_num)
        for a_category in FigureManager._figures_by_category.keys():
            if fig_num in FigureManager._figures_by_category[a_category]:
                FigureManager._figures_by_category[a_category].remove(fig_num)
        FigureManager._figures_by_category[category].append(fig_num)
        if make_current:
            FigureManager._category_active_figures[category] = fig_num

    @staticmethod
    def figure_closed(figure_number):
        if FigureManager._active_figure == figure_number:
            FigureManager._active_figure = NO_FIGURE
        for a_category in FigureManager._figures_by_category.keys():
            if figure_number in FigureManager._figures_by_category[a_category]:
                FigureManager._figures_by_category[a_category].remove(figure_number)
            if FigureManager._category_active_figures[a_category] == figure_number:
                FigureManager._category_active_figures[a_category] = NO_FIGURE
            if FigureManager._category_current_figures[a_category] == figure_number:
                FigureManager._category_active_figures[a_category] = NO_FIGURE
        try:
            del FigureManager._figures[figure_number]
        except KeyError:
            raise KeyError('The key "%s" does not exist. The figure cannot be closed')

    @staticmethod
    def get_category(figure_number):
        for category,fig_list in FigureManager._figures_by_category.items():
            if figure_number in fig_list:
                figure_category = category
                break
        else:
            raise KeyError("Figure no. %i was not found in any category "%figure_number if figure_number else 0)
            # in-line if handles the case figure_number is None
        return figure_category

    @staticmethod
    def set_figure_as_kept(figure_number):
        # kept _figures are just lying around, not really managed much, until they report in as current/active again
        figure_category = FigureManager.get_category(figure_number)

        if FigureManager._category_active_figures[figure_category] == figure_number:
            FigureManager._category_active_figures[figure_category] = NO_FIGURE

        if FigureManager._category_current_figures[figure_category] == figure_number:
            FigureManager._category_current_figures[figure_category] = NO_FIGURE

        FigureManager.broadcast(figure_category)

    @staticmethod
    def set_figure_as_current(figure_number):
        figure_category = FigureManager.get_category(figure_number)
        FigureManager._category_current_figures[figure_category] = figure_number
        FigureManager._category_active_figures[figure_category] = figure_number
        FigureManager._active_figure = figure_number
        FigureManager.broadcast()

    @staticmethod
    def set_figure_as_active(figure_number):
        figure_category = FigureManager.get_category(figure_number)
        if figure_number == FigureManager._category_current_figures[figure_category]:
            # Don't turn a current figure into an active figure.. doesnt make sense
            return
        FigureManager._category_active_figures[figure_category] = figure_number
        FigureManager._active_figure = figure_number

    @staticmethod
    def broadcast(category=None):
        if category is None:
            broadcast_list = FigureManager._figures_by_category.keys()
        else:
            broadcast_list = [category]

        for category in broadcast_list:
            for figure_number in FigureManager._figures_by_category[category]:

                if figure_number == FigureManager._category_current_figures[category]:
                    FigureManager._figures[figure_number].set_as_current()

                elif figure_number == FigureManager._category_active_figures[category]:
                    FigureManager._figures[figure_number].set_as_active()

                else:
                    FigureManager._figures[figure_number].set_as_kept()

    @staticmethod
    def all_figure_numbers():
        """An iterator over all figure numbers"""
        return FigureManager._figures.keys()

    @staticmethod
    def all_figures_numbers_in_category(category):
        """Return an iterator over all _figures numbers in a category"""
        return iter(FigureManager._figures_by_category[category])

    @staticmethod
    def unclassified_figures():
        """Return an iterator over all unclassified _figures"""
        return iter(FigureManager._unclassified_figures)

    @staticmethod
    def reset():
        """Reset all class variables to initial state. This Function exists for testing purposes """

        FigureManager._active_category = None
        FigureManager._category_active_figures = {"1d": NO_FIGURE, "2d": NO_FIGURE}  #active _figures recieve commands
        FigureManager._category_current_figures = {"1d": NO_FIGURE, "2d": NO_FIGURE}  # Current _figures are overplotted
        FigureManager._figures_by_category = {"1d": [], "2d": []}
        FigureManager._unclassified_figures = []
        FigureManager._active_figure = NO_FIGURE
        FigureManager._figures = {}

    @staticmethod
    def all_figures():
        """Return an iterator over all figures"""
        return FigureManager._figures.values()

    @staticmethod
    def number_of_figure(figure):
        for key,value in FigureManager._figures.items():
            if value == figure:
                return key
        raise ValueError('Figure %s was not recognised'%figure)

#This a decorator that accepts a parameter
def activate_category(category):

    def real_activate_function_decorator(function):

        def wrapper(*args, **kwargs):
            FigureManager._activate_category(category)
            return_value = function(*args, **kwargs)
            FigureManager._deactivate_category()
            return return_value
        wrapper.__name__ = function.__name__
        return wrapper

    return real_activate_function_decorator
