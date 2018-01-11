from matplotlib import text
from mslice.plotting.plot_window.quick_options import QuickAxisOptions, QuickLabelOptions, QuickLineOptions

def quick_options(target, model, log=None):
    if isinstance(target, text.Text):
        view = QuickLabelOptions(target)
        return QuickLabelPresenter(view, target, model)
    elif isinstance(target, str):
        if target[:1] == 'x' or target[:1] == 'y':
            grid = getattr(model, target[:-5] + 'grid')
        else:
            grid = None
        view = QuickAxisOptions(target, getattr(model, target), grid, log)
        return QuickAxisPresenter(view, target, model, grid, log)
    else:
        view = QuickLineOptions(model.get_line_data(target))
        return QuickLinePresenter(view, target, model)


class QuickAxisPresenter(object):

    def __init__(self, view, target, model, grid, log):
        self.view = view
        self.type = type
        self.model = model
        accepted = self.view.exec_()
        if accepted:
            self.set_range(target, log)
            if grid is not None:
                self.set_grid(target)

    def set_range(self, target, log):
        range = (float(self.view.range_min), float(self.view.range_max))
        setattr(self.model, target, range)
        if log is not None:
            setattr(self.model, target[:-5] + 'log', self.view.log_scale.isChecked())

    def set_grid(self, target):
        setattr(self.model, target[:-5] + 'grid', self.view.grid_state)

class QuickLabelPresenter(object):

    def __init__(self, view, target, model):
        self.view = view
        self.target = target
        self.model = model
        accepted = self.view.exec_()
        if accepted:
            self.set_label()

    def set_label(self):
        self.target.set_text(self.view.label)


class QuickLinePresenter(object):

    def __init__(self, view, target, model):
        self.view = view
        self.target = target
        self.model = model
        accepted = self.view.exec_()
        if accepted:
            self.set_line_options(target)

    def set_line_options(self, line):
        line_options = {}
        values = ['color', 'style', 'width', 'marker', 'label', 'shown', 'legend']
        for value in values:
            line_options[value] = getattr(self.view, value)
        self.model.set_line_data(line, line_options)
