from matplotlib import text
from matplotlib.mathtext import MathTextParser
from mslice.plotting.plot_window.quick_options import (
    QuickAxisOptions,
    QuickLabelOptions,
    QuickLineOptions,
    QuickError,
)


def quick_options(target, model, has_logarithmic=None, redraw_signal=None):
    """Find which quick_options to use based on type of target"""
    if isinstance(target, text.Text):
        quick_label_options(model.plot_window, target, redraw_signal)
    elif isinstance(target, str):
        return quick_axis_options(
            model.plot_window, target, model, has_logarithmic, redraw_signal
        )
    else:
        quick_line_options(model.plot_window, target, model)


def quick_label_options(parent, target, redraw_signal=None):
    view = QuickLabelOptions(parent, target, redraw_signal)
    view.ok_clicked.connect(lambda: _set_label_options(view, target))
    view.show()
    return view


def quick_axis_options(parent, target, model, has_logarithmic=None, redraw_signal=None):
    if target[:1] == "x" or target[:1] == "y":
        grid = getattr(model, target[:-5] + "grid")
    else:
        grid = None
    view = QuickAxisOptions(
        parent,
        target,
        getattr(model, target),
        getattr(model, target + "_font_size"),
        grid,
        has_logarithmic,
        redraw_signal,
    )
    view.ok_clicked.connect(
        lambda: _set_axis_options(view, target, model, has_logarithmic, grid)
    )
    view.show()
    return view


def quick_line_options(parent, target, model):
    view = QuickLineOptions(parent, model.get_line_options(target), model.show_legends)
    _run_quick_options(view, _set_line_options, model, target)


def _run_quick_options(view, update_model_function, *args):
    accepted = view.exec_()
    if accepted:
        update_model_function(view, *args)


def _set_axis_options(view, target, model, has_logarithmic, grid):
    if has_logarithmic is not None:
        setattr(model, target[:-5] + "log", view.log_scale.isChecked())
    if grid is not None:
        setattr(model, target[:-5] + "grid", view.grid_state)

    range = (float(view.range_min), float(view.range_max))
    setattr(model, target, range)

    setattr(model, target + "_font_size", view.font_size.value())


def _set_label_options(view, target):
    _set_label(view, target)
    _set_font_size(view, target)


def check_latex(value):
    if "$" in value:
        parser = MathTextParser("agg")
        try:
            parser.parse(value)
        except ValueError:
            return False
    return True


def _set_label(view, target):
    label = view.label
    if check_latex(label):
        target.set_text(label)
    else:
        QuickError("Invalid LaTeX in label string")


def _set_font_size(view, target):
    size = view.label_font_size
    target.set_size(size)


def _set_line_options(view, model, line):
    line_options = {}
    values = [
        "error_bar",
        "color",
        "style",
        "width",
        "marker",
        "label",
        "shown",
        "legend",
    ]
    for value in values:
        line_options[value] = getattr(view, value)
    model.set_line_options(line, line_options)
