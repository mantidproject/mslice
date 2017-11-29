from functools import partial

from matplotlib import lines, legend
from mslice.plotting.plot_window.quick_options import QuickLineOptions

def quick_options(target, model):
    if isinstance(target, str):
        view = quick_options_from_str(target)
    else:
        view = QuickLineOptions(target)
        return QuickLinePresenter(view, target, model)


class QuickLinePresenter(object):

    def __init__(self, view, target, model):
        self.view = view
        self.target = target
        self.model = model
        self.view.ok_clicked.connect(partial(self.set_line_options, target))
        self.view.cancel_clicked.connect(self.close)

    def set_line_options(self, line):
        print("OK!")
        line.set_color(self.view.color)
        line.set_linestyle(self.view.style)
        line.set_linewidth(self.view.width)
        line.set_marker(self.view.marker)
        line.set_label(self.view.label)
        if not self.view.shown:
            line.set_linestyle('')
        if not self.view.legend:
            line.set_label('')
        self.model.reset_info_checkboxes()
        self.model.update_slice_legend()
        self.model.canvas.draw()
        self.close()

    def close(self):
        self.view.close()

