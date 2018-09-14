from six import string_types
from matplotlib import text
from mslice.plotting.plot_window.quick_options import QuickAxisOptions, QuickLabelOptions, QuickLineOptions


def quick_options(target, model, has_logarithmic=None):
    """Find which quick_options to use based on type of target"""
    if isinstance(target, text.Text):
        quick_label_options(target)
    elif isinstance(target, string_types):
        quick_axis_options(target, model, has_logarithmic)
    else:
        quick_line_options(target, model)


def quick_label_options(target):
    view = QuickLabelOptions(target)
    _run_quick_options(view, _set_label, target)


def quick_axis_options(target, model, has_logarithmic=None):
    if target[:1] == 'x' or target[:1] == 'y':
        grid = getattr(model, target[:-5] + 'grid')
    else:
        grid = None
    view = QuickAxisOptions(target, getattr(model, target), grid, has_logarithmic)
    _run_quick_options(view, _set_axis_options, target, model, has_logarithmic, grid)


def quick_line_options(target, model):
    view = QuickLineOptions(model.get_line_options(target))
    _run_quick_options(view, _set_line_options, model, target)


def _run_quick_options(view, update_model_function, *args):
    accepted = view.exec_()
    if accepted:
        update_model_function(view, *args)


def _set_axis_options(view, target, model, has_logarithmic, grid):
    range = (float(view.range_min), float(view.range_max))
    setattr(model, target, range)
    if has_logarithmic is not None:
        setattr(model, target[:-5] + 'log', view.log_scale.isChecked())
    if grid is not None:
        setattr(model, target[:-5] + 'grid', view.grid_state)


def _set_label(view, target):
    target.set_text(view.label)


def _set_line_options(view, model, line):
    line_options = {}
    values = ['error_bar', 'color', 'style', 'width', 'marker', 'label', 'shown', 'legend']
    for value in values:
        line_options[value] = getattr(view, value)
    model.set_line_options(line, line_options)
