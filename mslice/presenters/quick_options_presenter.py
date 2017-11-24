from functools import partial

from matplotlib import lines, legend
from mslice.plotting.plot_window.quick_options import QuickLineOptions

def quick_options(target, model):
    if isinstance(target, str):
        view = quick_options_from_str(target)
    if isinstance(target, lines.Line2D):
        view = QuickLineOptions(target)
        QuickLinePresenter(view, target, model)
    elif isinstance(target, legend.Legend):
        view = createLegendOptions(target)

class QuickLinePresenter(object):
    def __init__(self, view, target, model):
        self.view = view
        self.target = target
        self.model = model
        self.view.ok_button.clicked.connect(partial(self.set_line_options, target))

    def set_line_options(self, line):
        print("OK!")
        line.set_color(self.view.color)
        line.set_linestyle(self.view.style)
        line.set_linewidth(self.view.width)
        line.set_marker(self.view.marker)
        self.model.canvas.draw()