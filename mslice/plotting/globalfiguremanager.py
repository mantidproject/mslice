"""
This module is responsible for handling which figures are current (will receive next plot operation) and which are
kept (will not be modified until it is made current once again).

The FigureManager in its responsibilities highly resembles the class maptplotlib._pylab_helpers.Gcf, However it
adds the functionality of having multiple 'categories' with each category having it own current window.
This is achieved through use of the supplied decorator 'activate_category' This decorator accepts one parameter  (a
string ) specifying which category a function belongs two. For instance to apply the decorator
activate_category('<category>') to the the function pyplot.pcolor would signal that the function pcolor would only apply
to  plots of the category '<category>'. All of this is done through manipulating the return value of `pyplot.gcf`. gcf
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
# system imports
from functools import wraps

import atexit

from enum import Enum
from .observabledictionary import DictionaryAction, ObservableDictionary

# Labels for each category
CATEGORY_CUT, CATEGORY_SLICE = "1d", "2d"


class FigureAction(Enum):
    Update = 0
    New = 1
    Closed = 2
    Renamed = 3
    OrderChanged = 4
    VisibilityChanged = 5


class GlobalFigureManagerObserver(object):
    def __init__(self, figure_manager=None):
        """
        :param figure_manager: Figure manager that will be used to notify observers.
                               Injected on initialisation for easy mocking
        """
        self.figure_manager = figure_manager

    def notify(self, action, key):
        """
        This method is called when a dictionary entry is added,
        removed or changed
        :param action: An enum with the type of dictionary action
        :param key: The key in the dictionary that was changed
        :param old_value: Old value(s) removed
        """

        if action == DictionaryAction.Create:
            self.figure_manager.notify_observers(FigureAction.New, key)
        elif action == DictionaryAction.Set:
            self.figure_manager.notify_observers(FigureAction.Renamed, key)
        elif action == DictionaryAction.Removed:
            self.figure_manager.notify_observers(FigureAction.Closed, key)
        elif action == DictionaryAction.Update:
            self.figure_manager.notify_observers(FigureAction.Update, key)
        elif action == DictionaryAction.Clear:
            # On Clear notify the observers to close all of the figures
            # `figs.keys()` is safe to iterate and delete items at the same time
            # because `keys` returns a new list, not referencing the original dict
            for key in self.figure_manager.figs.keys():
                self.figure_manager.notify_observers(FigureAction.Closed, key)
        else:
            raise ValueError("Notifying for action {} is not supported".format(action))


class GlobalFigureManager(object):
    """Static class to manage a set of numbered figures.

    It is never instantiated. It consists of attributes to
    manage a set of "current" figures along with keeping
    track of all created figures. Each current figure is expected
    to be placed into a category such that a current figure for
    a given category can be returned to be operated on separately
    to the current figure for another category.

            *_activeQue*:
          list of *managers*, with active one at the end

    """
    # if there is a current figure it should be both current and active
    _active_category = None
    _category_current_figures = {
        CATEGORY_CUT: None,
        CATEGORY_SLICE: None
    }  # Current_figures receive decorated commands
    _figures_by_category = {CATEGORY_CUT: [], CATEGORY_SLICE: []}
    _unclassified_figures = []
    _active_figure = None
    _last_active_figure = None
    _figures = {}
    # If the following attribute is True, "Make Current" will be disabled for all open figures. This will be used for
    # the interactive cut window to stop other windows being made current whilst the interactive cut is active.
    _disable_make_current = False

    _activeQue = []
    figs = ObservableDictionary({})
    observers = []

    @classmethod
    def initialiseFiguresObserver(cls):
        """
        This is used to inject the GlobalFigureManager into the GlobalFigureManagerObserver
        as there is no way to reference the class' own name inside the class' own definition
        :return:
        """
        cls.figs.add_observer(GlobalFigureManagerObserver(cls))

    @classmethod
    def destroy(cls, num):
        """
        Try to remove all traces of figure *num*.

        In the interactive backends, this is bound to the
        window "destroy" and "delete" events.
        """
        if not cls.has_fignum(num):
            return
        if num in cls._unclassified_figures:
            cls._unclassified_figures.remove(num)
            return
        if cls._active_figure == num:
            cls._active_figure = None
        try:
            category = cls.get_category(num)
            if cls._category_current_figures[category] == num:
                cls._category_current_figures[category] = None
            cls._figures_by_category[category].remove(num)
        except KeyError:
            pass
        current_fig_manager = cls._figures[num]
        cls._remove_manager_if_present(current_fig_manager)
        del cls._figures[num]
        current_fig_manager.destroy()
        cls.notify_observers(FigureAction.Closed, num)

    @classmethod
    def get_figure_by_number(cls, num):
        """
        Returns the figure with figure number num
        :param num: The assigned figure number
        :return: The figure with figure number num
        """
        if cls.has_fignum(num):
            return cls._figures[num]
        else:
            return None

    @classmethod
    def destroy_fig(cls, fig):
        """*fig* is a Figure instance"""
        num = next((manager.num for manager in cls._figures.values()
                    if manager.canvas.figure == fig), None)
        if num is not None:
            cls.destroy(num)

    @classmethod
    def destroy_all(cls):
        # this is need to ensure that gc is available in corner cases
        # where modules are being torn down after install with easy_install
        import gc  # noqa
        for manager in list(cls._figures.values()):
            manager.destroy()
        cls._active_figure = None
        cls._category_current_figures = {
            CATEGORY_CUT: None,
            CATEGORY_SLICE: None
        }
        cls._activeQue = []
        cls._figures.clear()

    @classmethod
    def has_fignum(cls, num):
        """
        Return *True* if figure *num* exists.
        """
        return num in cls._figures

    @classmethod
    def get_all_fig_managers(cls):
        """
        Return a list of figure managers.
        """
        return list(cls._figures.values())

    @classmethod
    def get_active(cls):
        """
        Return the manager of the active figure, or *None*.
        """
        if len(cls._activeQue) == 0:
            return None
        else:
            return cls._activeQue[-1]

    @classmethod
    def set_active(cls, manager):
        """
        Make the figure corresponding to *manager* the active one.
        """
        cls._remove_manager_if_present(manager)
        cls._activeQue.append(manager)
        cls.figs[manager.num] = manager
        cls.notify_observers(FigureAction.OrderChanged, manager.num)

    @classmethod
    def _remove_manager_if_present(cls, manager):
        """
        Removes the manager from the active queue, if it is present in it.
        :param manager: Manager to be removed from the active queue
        :return:
        """
        try:
            del cls._activeQue[cls._activeQue.index(manager)]
        except ValueError:
            # the figure manager was not in the active queue - no need to delete anything
            pass

    @classmethod
    def draw_all(cls, force=False):
        """
        Redraw all figures registered with the pyplot
        state machine.
        """
        for f_mgr in cls.get_all_fig_managers():
            if force or f_mgr.canvas.figure.stale:
                f_mgr.canvas.draw_idle()

    # ------------------ Our additional interface -----------------

    @classmethod
    def last_active_values(cls):
        """
        Returns a dictionary where the keys are the plot numbers and
        the values are the last shown (active) order, the most recent
        being 1, the oldest being N, where N is the number of figure
        managers
        :return: A dictionary with the values as plot number and keys
                 as the opening order
        """
        last_shown_order_dict = {}
        num_figure_managers = len(cls._activeQue)

        for index in range(num_figure_managers):
            last_shown_order_dict[cls._activeQue[index].num] = num_figure_managers - index

        return last_shown_order_dict

    @classmethod
    def reset(cls):
        """Reset all class variables to initial state. This function exists for testing purposes """
        cls._active_category = None
        cls._category_current_figures = {
            CATEGORY_CUT: None,
            CATEGORY_SLICE: None
        }  # Current _figures are overplotted
        cls._figures_by_category = {CATEGORY_CUT: [], CATEGORY_SLICE: []}
        cls._unclassified_figures = []
        cls._active_figure = None
        cls._figures = {}

    @classmethod
    def _new_figure(cls, num=None):
        # local import to avoid circular dependency in figure_manager_test.py
        # mock.patch can't patch a class where the module has already been imported
        from mslice.plotting.plot_window.plot_figure_manager import new_plot_figure_manager
        if num is None:
            num = 1
            while any([
                    num == existing_fig_num
                    for existing_fig_num in cls._figures.keys()
            ]):
                num += 1
        new_fig = new_plot_figure_manager(num, GlobalFigureManager)
        cls._figures[num] = new_fig
        cls._active_figure = num
        cls._unclassified_figures.append(num)
        cls.notify_observers(FigureAction.New, num)
        cls.notify_observers(FigureAction.Renamed, num)
        return cls._figures[num], num

    @classmethod
    def get_figure_number(cls, num=None, create_if_not_found=True):
        cls._active_figure = num
        try:
            # make the figure the current figure of its category
            category = cls.get_category(num)
            cls._category_current_figures[category] = num
        except KeyError:
            # the figure is still uncategorised, Do nothing
            pass

        figure = cls._figures.get(num, None)
        if figure or not create_if_not_found:
            return figure
        else:
            # return the figure discard the number
            return cls._new_figure(num)[0]

    @classmethod
    def get_fig_manager(cls, num=None, create_if_not_found=True):
        return cls.get_figure_number(num, create_if_not_found)

    @classmethod
    def get_active_figure(cls):
        if cls._active_category:
            if cls._active_figure in cls._unclassified_figures:
                cls.assign_figure_to_category(cls._active_figure,
                                              cls._active_category,
                                              make_current=True)
            elif cls._category_current_figures[cls._active_category] is None:
                _, num = cls._new_figure()
                cls.assign_figure_to_category(num,
                                              cls._active_category,
                                              make_current=True)
                cls._active_figure = num
            else:
                cls._active_figure = cls._category_current_figures[
                    cls._active_category]
        else:
            if cls._active_figure is None:
                fig, num = cls._new_figure()
                cls._active_figure = num
        cls.notify_observers(FigureAction.Renamed, cls._active_figure)
        return cls._figures[cls._active_figure]

    @classmethod
    def active_cut_figure_exists(cls):
        return cls._category_current_figures[CATEGORY_CUT] is not None

    @classmethod
    def activate_category(cls, category):
        """Sets the active category to the supplied argument. Do not call this function directly, instead use supplied
        decorator below 'activate_category' """
        cls._active_category = category

    @classmethod
    def deactivate_category(cls):
        """ Unsets the active category. Do not call this function directly, instead use supplied decorator
        below 'activate_category' """
        cls._active_category = None

    @classmethod
    def assign_figure_to_category(cls, num, category, make_current=False):
        if num not in cls._figures:
            raise ValueError("Figure does not exist")

        if num in cls._unclassified_figures:
            cls._unclassified_figures.remove(num)

        for a_category in cls._figures_by_category:
            if num in cls._figures_by_category[a_category]:
                cls._figures_by_category[a_category].remove(num)
            if cls._category_current_figures == num:
                cls._category_current_figures = None

        cls._figures_by_category[category].append(num)
        if make_current:
            cls._category_current_figures[category] = num
        cls.broadcast()

    @classmethod
    def figure_closed(cls, num):
        """Figure is closed, remove all references to it from all internal list

        If it was the category current or global active figure then set that to None"""
        if cls._active_figure == num:
            cls._active_figure = None
        for a_category in cls._figures_by_category:
            if num in cls._figures_by_category[a_category]:
                cls._figures_by_category[a_category].remove(num)

            if cls._category_current_figures[a_category] == num:
                cls._category_current_figures[a_category] = None
        try:
            del cls._figures[num]
        except KeyError:
            raise KeyError(
                'The key "%s" does not exist. The figure cannot be closed' %
                num)

    @classmethod
    def get_category(cls, num):
        """Return the category of the figure"""
        for category, fig_list in list(cls._figures_by_category.items()):
            if num in fig_list:
                figure_category = category
                break
        else:
            raise KeyError("Figure no. %i was not found in any category " %
                           num if num else 0)
            # in-line if handles the case num is None
        return figure_category

    @classmethod
    def set_figure_as_kept(cls, num=None):
        # kept figures are just lying around, not really managed much, until they report in as current again
        if num is None:
            if cls._active_figure is not None:
                num = cls.get_active_figure().number
                cls._last_active_figure = num
            else:
                num = cls._last_active_figure

        if cls._active_figure == num:
            cls._active_figure = None
        try:
            figure_category = cls.get_category(num)
        except KeyError:
            figure_category = None

        if figure_category:
            if cls._category_current_figures[figure_category] == num:
                cls._category_current_figures[figure_category] = None

        cls.broadcast(figure_category)

    @classmethod
    def set_figure_as_current(cls, num=None):
        if cls._disable_make_current:
            cls.broadcast(None)
            return

        if num is None:
            if cls._last_active_figure is None:
                num = cls.get_active_figure().number
            else:
                num = cls._last_active_figure

        try:
            figure_category = cls.get_category(num)
        except KeyError:
            figure_category = None

        if figure_category:
            cls._category_current_figures[figure_category] = num

        cls._active_figure = num
        cls.broadcast(figure_category)

    @classmethod
    def broadcast(cls, category=None):
        """This method will broadcast to all figures in 'category' to update the displayed kept/current status"""
        if category is None:
            broadcast_list = cls._figures_by_category
        else:
            broadcast_list = [category]

        for category in broadcast_list:
            for figure_number in cls._figures_by_category[category]:

                if cls._category_current_figures[category] == figure_number:
                    cls._figures[figure_number].flag_as_current()

                else:
                    cls._figures[figure_number].flag_as_kept()

        for figure in cls._unclassified_figures:
            if figure == cls._active_figure:
                cls._figures[figure].flag_as_current()
            else:
                cls._figures[figure].flag_as_kept()

    @classmethod
    def disable_make_current(cls, category=None):
        cls._disable_make_current = True

    @classmethod
    def enable_make_current(cls, category=None):
        cls._disable_make_current = False

    @classmethod
    def make_current_disabled(cls):
        return cls._disable_make_current

    @classmethod
    def all_figure_numbers(cls):
        """An iterator over all figure numbers"""
        return list(cls._figures.keys())

    @classmethod
    def all_figures_numbers_in_category(cls, category):
        """Return an iterator over all _figures numbers in a category"""
        return iter(cls._figures_by_category[category])

    @classmethod
    def unclassified_figures(cls):
        """Return an iterator over all unclassified _figures"""
        return iter(cls._unclassified_figures)

    @classmethod
    def all_figures(cls):
        """Return an iterator over all figures"""
        return list(cls._figures.values())

    @classmethod
    def number_of_figure(cls, fig):
        for key, value in list(cls._figures.items()):
            if value == fig:
                return key
        raise ValueError('Figure %s was not recognised' % fig)

    # ---------------------- Observer methods ---------------------
    # This is currently very simple as the only observer is
    # permanently registered to this class.

    @classmethod
    def add_observer(cls, observer):
        """
        Add an observer to this class - this can be any class with a
        notify() method
        :param observer: A class with a notify method
        """
        assert "notify" in dir(observer), "An observer must have a notify method"
        cls.observers.append(observer)

    @classmethod
    def notify_observers(cls, action, figure_number):
        """
        Calls notify method on all observers
        :param action: A FigureAction enum for the action called
        :param figure_number: The unique fig number (key in the dict)
        """
        for observer in cls.observers:
            observer.notify(action, figure_number)

    @classmethod
    def figure_title_changed(cls, figure_number):
        """
        Notify the observers that a figure title was changed
        :param figure_number: The unique number in GlobalFigureManager
        """
        cls.notify_observers(FigureAction.Renamed, figure_number)

    @classmethod
    def figure_visibility_changed(cls, figure_number):
        """
        Notify the observers that a figure was shown or hidden
        :param figure_number: The unique number in GlobalFigureManager
        """
        cls.notify_observers(FigureAction.VisibilityChanged, figure_number)


def set_category(category):
    """A decorator to mark a function as part of the given category. For details
    of the category mechanism see the docstring on the currentfigure
    module.

    :param category: 'cut' or 'slice' to denote the category of the plot produced
    """

    def activate_impl(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            try:
                GlobalFigureManager.activate_category(category)
                return_value = function(*args, **kwargs)
            finally:
                GlobalFigureManager.deactivate_category()
            return return_value

        return wrapper

    return activate_impl


GlobalFigureManager.initialiseFiguresObserver()
atexit.register(GlobalFigureManager.destroy_all)
