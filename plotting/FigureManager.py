from plot_figure import PlotFigure
NO_FIGURE = -1723


class FigureManager(object):
    """This is singleton static class to manage the current figures

    Do not instantiate this class"""
    #if there is a current figure it should be both current and active

    _active_category = None
    _category_active_figures = {"1d": NO_FIGURE, "2d": NO_FIGURE} #active figures recieve commands
    _category_current_figures = {"1d": NO_FIGURE, "2d": NO_FIGURE}  # Current figures are overplotted
    _figures_by_category = {"1d": [], "2d": []}
    unclassified_figures = []
    _active_figure = NO_FIGURE
    figures = {}

    def __init__(self, *args, **kwargs):
        raise Exception("This is class is static class singleton. Do not Instantiate it")

    @staticmethod
    def new_figure(fig_num=None):
        if fig_num is None:
            fig_num = 1
            while any([fig_num == existing_fig_num for existing_fig_num in FigureManager.figures.keys()]):
                fig_num += 1
        new_fig = PlotFigure(fig_num,FigureManager)
        FigureManager.figures[fig_num] = new_fig
        FigureManager._active_figure = fig_num
        FigureManager.unclassified_figures.append(fig_num)
        return FigureManager.figures[fig_num],fig_num

    @staticmethod
    def get_figure_number(fig_num):
        FigureManager._active_figure = fig_num
        try:
            category = FigureManager.get_category(fig_num)
            FigureManager._category_active_figures[category] = fig_num
        except ValueError:
            #the figure is still uncategorised
            pass
        #TODO make it active for category
        figure = FigureManager.figures.get(fig_num, None)
        if figure:
            return figure
        else:
            # return the figure discard the number
            return FigureManager.new_figure(fig_num)[0]
    @staticmethod
    def get_active_figure():
        if FigureManager._active_category:
            if FigureManager._category_active_figures[FigureManager._active_category] == NO_FIGURE:
                fig,num = FigureManager.new_figure()
                FigureManager.assign_figure_to_category(num,FigureManager._active_category)
                FigureManager._category_active_figures[FigureManager._active_category] = num
                FigureManager._active_figure = num
            else:
                FigureManager._active_figure = FigureManager._category_active_figures[FigureManager._active_category]
        if FigureManager._active_figure == NO_FIGURE:
            FigureManager.new_figure()
        return FigureManager.figures[FigureManager._active_figure]

    @staticmethod
    def activate_category(category):
        FigureManager._active_category = category

    @staticmethod
    def deactivate_category():
        FigureManager._active_category = None

    @staticmethod
    def assign_figure_to_category(fig_num,category,make_current=False):
        if fig_num not in FigureManager.figures.keys():
            raise ValueError("Figure does not exist")
        if fig_num in FigureManager.unclassified_figures:
            FigureManager.unclassified_figures.remove(fig_num)
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

    @staticmethod
    def get_category(figure_number):
        for category,fig_list in FigureManager._figures_by_category.items():
            if figure_number in fig_list:
                figure_category = category
                break
        else:
            raise ValueError("Figure no. %i was not found"%figure_number)
        return figure_category

    @staticmethod
    def set_figure_as_kept(figure_number):
        # kept figures are just lying around, not really managed much, until they report in as current/active again
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
                    FigureManager.figures[figure_number].set_as_current()

                elif figure_number == FigureManager._category_active_figures[category]:
                    FigureManager.figures[figure_number].set_as_active()

                else:
                    FigureManager.figures[figure_number].set_as_kept()