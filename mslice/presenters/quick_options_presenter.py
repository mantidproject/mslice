from six import string_types
from matplotlib import text
from mslice.plotting.plot_window.quick_options import QuickAxisOptions, QuickLabelOptions, QuickLineOptions


def quick_options(target, model, has_logarithmic=None):
    if isinstance(target, text.Text):
        run_quick_options(QuickLabelOptions(target), set_label, target)
    elif isinstance(target, string_types):
        if target[:1] == 'x' or target[:1] == 'y':
            grid = getattr(model, target[:-5] + 'grid')
        else:
            grid = None
        view = QuickAxisOptions(target, getattr(model, target), grid, has_logarithmic)
        run_quick_options(view, set_axis_range, target, model, has_logarithmic, grid)
    else:
        view = QuickLineOptions(model.get_line_data(target))
        run_quick_options(view, set_line_options, model, target)


def run_quick_options(view, update_model_function, *args):
    accepted = view.exec_()
    if accepted:
        update_model_function(view, *args)


def set_axis_range(view, target, model, has_logarithmic, grid):
    range = (float(view.range_min), float(view.range_max))
    setattr(model, target, range)
    if has_logarithmic is not None:
        setattr(model, target[:-5] + 'log', view.log_scale.isChecked())
    if grid is not None:
        setattr(model, target[:-5] + 'grid', view.grid_state)


def set_label(view, target):
    target.set_text(view.label)


def set_line_options(view, model, line):
    line_options = {}
    values = ['color', 'style', 'width', 'marker', 'label', 'shown', 'legend']
    for value in values:
        line_options[value] = getattr(view, value)
    model.set_line_data(line, line_options)
