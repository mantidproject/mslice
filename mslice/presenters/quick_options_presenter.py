from matplotlib import text
from mslice.plotting.plot_window.quick_options import QuickAxisOptions, QuickLabelOptions, QuickLineOptions

def quick_options(target, model, log=None):
    if isinstance(target, text.Text):
        view = QuickLabelOptions(target)
        return QuickLabelPresenter(view, target, model)
    elif isinstance(target, str):
        view = QuickAxisOptions(target, getattr(model, target), log)
        return QuickAxisPresenter(view, target, model, log)
    else:
        view = QuickLineOptions(target)
        return QuickLinePresenter(view, target, model)


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
        line.set_color(self.view.color)
        line.set_linestyle(self.view.style)
        line.set_linewidth(self.view.width)
        line.set_marker(self.view.marker)
        line.set_label(self.view.label)
        if not self.view.shown:
            line.set_linestyle('')
        if not self.view.legend:
            line.set_label('')
