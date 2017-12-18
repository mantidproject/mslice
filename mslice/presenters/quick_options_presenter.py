from matplotlib import text
from matplotlib.legend import Legend
from mslice.plotting.plot_window.quick_options import QuickAllLineOptions, QuickAxisOptions, \
                                                      QuickLabelOptions, QuickLineOptions

def quick_options(target, model, log=None):
    if isinstance(target, text.Text):
        view = QuickLabelOptions(target)
        return QuickLabelPresenter(view, target, model)
    elif isinstance(target, str):
        view = QuickAxisOptions(target, getattr(model, target), log)
        return QuickAxisPresenter(view, target, model, log)
    elif isinstance(target, Legend):
        view = QuickAllLineOptions(model.get_all_line_data())
        return QuickAllLinePresenter(view, model)
    else:
        view = QuickLineOptions(model.get_line_data(target))
        return QuickLinePresenter(view, target, model)


def properties():
    return ['color', 'style', 'width', 'marker', 'label', 'shown', 'legend']


class QuickAxisPresenter(object):

    def __init__(self, view, target, model, log):
        self.view = view
        self.type = type
        self.model = model
        accepted = self.view.exec_()
        if accepted:
            self.set_range(target, log)

    def set_range(self, target, log):
        range = (float(self.view.range_min), float(self.view.range_max))
        setattr(self.model, target, range)
        if log is not None:
            setattr(self.model, target[:-5] + 'log', self.view.log_scale.isChecked())


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
        for p in properties():
            line_options[p] = getattr(self.view, p)
        self.model.set_line_data(line, line_options)

class QuickAllLinePresenter(object):

    def __init__(self, view, model):
        self.view = view
        self.model = model
        accepted = self.view.exec_()
        if accepted:
            self.set_line_options()

    def set_line_options(self):
        all_line_options = []
        line_options = {}
        for line_widget in self.view.line_widgets:
            for p in properties():
                line_options[p] = getattr(line_widget, p)
            all_line_options.append(line_options)
        self.model.set_all_line_data(all_line_options)
