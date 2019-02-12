from datetime import datetime
from mslice.cli.helperfunctions import _function_to_intensity
from mslice.models.labels import get_recoil_key

COMMON_PACKAGES = ["import mslice.cli as mc", "import mslice.plotting.pyplot as plt"]
MPL_COLORS_IMPORT = ["\nimport matplotlib.colors as colors"]
NUMPY_IMPORT = ["\nimport numpy as np"]
LOG_SCALE_MIN = 0.001


def cleanup(script_lines):
    """
    Removes data preprocessing lines from the workspace history script.
    i.e All lines before the script was first loaded
    """
    for line in script_lines[-1:]:
        if line.startswith("Load"):
            index = script_lines.index(line)
            return script_lines[index:]


def header(plot_handler):
    """Creates a list of import statements to be used in the generated script header"""
    from mslice.plotting.plot_window.cut_plot import CutPlot
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    statements = list("# Python Script Generated by Mslice on {}\n".format(datetime.now().replace(microsecond=0)))

    statements.append("\n".join(COMMON_PACKAGES))
    if isinstance(plot_handler, SlicePlot) and plot_handler.colorbar_log is True:
        statements.append("\n".join(MPL_COLORS_IMPORT))
    if isinstance(plot_handler, CutPlot) and (plot_handler.x_log is True or plot_handler.y_log is True):
        statements.append("\n".join(NUMPY_IMPORT))

    return statements


def add_header(script_lines, plot_handler):
    """Adds header to script lines"""
    for i, statement in enumerate(header(plot_handler)):
        script_lines.insert(i, statement)


def add_plot_statements(script_lines, plot_handler):
    """Adds plot statements to the script lines used to generate the python script"""
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    from mslice.plotting.plot_window.cut_plot import CutPlot

    add_header(script_lines, plot_handler)

    line_no = len(script_lines)
    script_lines.insert(line_no - 1, "\n")
    script_lines[line_no] = "ws = mc.{}\n".format(script_lines[line_no])

    if plot_handler is not None:
        if isinstance(plot_handler, SlicePlot):
            add_slice_plot_statements(script_lines, plot_handler)
            add_overplot_statements(script_lines, plot_handler)
        elif isinstance(plot_handler, CutPlot):
            add_cut_plot_statements(script_lines, plot_handler)

        script_lines.append("mc.Show()\n")

    return script_lines


def add_slice_plot_statements(script_lines, plot_handler):
    default_opts = plot_handler.default_options
    cache = plot_handler._slice_plotter_presenter._slice_cache

    slice = cache[plot_handler.ws_name]
    momentum_axis = str(slice.momentum_axis)
    energy_axis = str(slice.energy_axis)
    norm = slice.norm_to_one

    script_lines.append('slice_ws = mc.Slice(ws, Axis1=\'{}\', Axis2=\'{}\', NormToOne={})\n\n'.format(
        momentum_axis, energy_axis, norm))
    script_lines.append('fig = plt.gcf()\n')
    script_lines.append('ax = fig.add_subplot(111, projection=\'mslice\')\n')

    if default_opts['intensity'] is True:
        intensity = _function_to_intensity[default_opts['intensity_method']]
        if default_opts['temp_dependent']:
            script_lines.append(
                'mesh = ax.pcolormesh(slice_ws, cmap=\'{}\', intensity=\'{}\', temperature={})\n'.format(
                    cache[plot_handler.ws_name].colourmap, intensity,
                    default_opts['temp']))
        else:
            script_lines.append('mesh = ax.pcolormesh(slice_ws, cmap=\'{}\', intensity=\'{}\')\n'.format(
                cache[plot_handler.ws_name].colourmap, intensity))
    else:
        script_lines.append('mesh = ax.pcolormesh(slice_ws, cmap=\'{}\')\n'.format(
            cache[plot_handler.ws_name].colourmap))

    script_lines.append("mesh.set_clim({}, {})\n".format(*plot_handler.colorbar_range))
    if plot_handler.colorbar_log:
        min, max = plot_handler.colorbar_range[0], plot_handler.colorbar_range[1]
        min = max(min, LOG_SCALE_MIN)
        script_lines.append("mesh.set_norm(colors.LogNorm({}, {}))\n".format(min, max))

    script_lines.append("cb = plt.colorbar(mesh, ax=ax)\n")
    script_lines.append("cb.set_label('{}', labelpad=20, rotation=270, picker=5)\n".format(plot_handler.colorbar_label))
    add_plot_options(script_lines, plot_handler)


def add_overplot_statements(script_lines, plot_handler):
    """Adds overplot line statements to the script if they were plotted"""
    ax = plot_handler._canvas.figure.gca()
    line_artists = ax.lines

    for line in line_artists:
        label = line._label
        if "nolegend" in label:
            continue
        key = get_recoil_key(label)
        rmm = int(label.split()[-1]) if "Relative" in label else None
        element = label if rmm is None else None
        recoil = True if rmm is not None or key in [1, 2, 4] else False
        cif = None  # Does not yet account for CIF files

        if recoil:
            if element is None:
                script_lines.append(
                    "ax.recoil(workspace='{}', rmm={})\n".format(
                        plot_handler.ws_name, rmm))
            else:
                script_lines.append(
                    "ax.recoil(workspace='{}', element='{}')\n".format(
                        plot_handler.ws_name, element))
        else:
            if cif is None:
                script_lines.append(
                    "ax.bragg(workspace='{}', element='{}')\n".format(
                        plot_handler.ws_name, element))

            else:
                script_lines.append(
                    "ax.bragg(workspace='{}', cif='{}')\n".format(
                        plot_handler.ws_name, cif))


def add_cut_plot_statements(script_lines, plot_handler):
    """Adds cut specific statements to the script"""
    default_opts = plot_handler.default_options
    script_lines.append('fig = plt.gcf()\n')
    script_lines.append('ax = fig.add_subplot(111, projection=\'mslice\')\n\n')

    add_cut_lines(script_lines, plot_handler)
    add_plot_options(script_lines, plot_handler)

    script_lines.append("ax.set_xscale('symlog', linthreshx=pow(10, np.floor(np.log10({}))))\n".format(
        default_opts["xmin"]) if plot_handler.is_changed("x_log") else "")

    script_lines.append("ax.set_yscale('symlog', linthreshx=pow(10, np.floor(np.log10({}))))\n".format(
        default_opts["ymin"]) if plot_handler.is_changed("x_log") else "")


def add_cut_lines(script_lines, plot_handler):
    cuts = plot_handler._cut_plotter_presenter._cut_cache_list
    errorbars = plot_handler._canvas.figure.gca().containers
    add_cut_lines_with_width(errorbars, script_lines, cuts)


def add_cut_lines_with_width(errorbars, script_lines, cuts):
    """Adds the cut statements for each interval of the cuts that were plotted"""
    i = 0
    for cut in cuts:
        integration_start = cut.integration_axis.start
        integration_end = cut.integration_axis.end
        cut_start, cut_end = integration_start, min(integration_start + cut.width, integration_end)
        intensity_range = (cut.intensity_start, cut.intensity_end)
        axis_units = cut.cut_axis.units
        norm_to_one = cut.norm_to_one

        while cut_start != cut_end and i < len(errorbars):
            cut.integration_axis.start = cut_start
            cut.integration_axis.end = cut_end
            cut_axis = str(cut.cut_axis)
            integration_axis = str(cut.integration_axis)

            script_lines.append('cut_ws_{} = mc.Cut(ws, CutAxis=\'{}\', IntegrationAxis=\'{}\', '
                                'NormToOne={})\n'.format(i, cut_axis, integration_axis, norm_to_one))

            errorbar = errorbars[i]
            colour = errorbar.lines[0]._color
            marker = errorbar.lines[0]._marker._marker
            style = errorbar.lines[0]._linestyle
            width = errorbar.lines[0]._linewidth
            label = errorbar._label

            if intensity_range != (None, None):
                script_lines.append(
                    'ax.errorbar(cut_ws_{}, x_units=\'{}\', label=\'{}\', color=\'{}\', marker=\'{}\', ls=\'{}\', '
                    'lw={}, intensity_range={})\n\n'.format(i, axis_units, label, colour, marker, style, width,
                                                            intensity_range))
            else:
                script_lines.append(
                    'ax.errorbar(cut_ws_{}, x_units=\'{}\', label=\'{}\', color=\'{}\', marker=\'{}\', ls=\'{}\', '
                    'lw={})\n\n'.format(i, axis_units, label, colour, marker, style, width))

            cut_start, cut_end = cut_end, min(cut_end + cut.width, integration_end)
            i += 1
        cut.reset_integration_axis(cut.start, cut.end)


def add_plot_options(script_lines, plot_handler):
    """Adds lines that change the plot options if they were modified"""
    script_lines.append("ax.set_title('{}')\n".format(plot_handler.title))

    if plot_handler.is_changed("y_label"):
        script_lines.append("ax.set_ylabel('{}')\n".format(plot_handler.y_label))

    if plot_handler.is_changed("x_label"):
        script_lines.append("ax.set_xlabel('{}')\n".format(plot_handler.x_label))

    if plot_handler.is_changed("y_grid"):
        script_lines.append("ax.grid({}, axis='y')\n".format(plot_handler.y_grid))

    if plot_handler.is_changed("x_grid"):
        script_lines.append("ax.grid({}, axis='x')\n".format(plot_handler.x_grid))

    if plot_handler.is_changed("y_range"):
        script_lines.append("ax.set_ylim(bottom={}, top={})\n".format(*plot_handler.y_range))

    if plot_handler.is_changed("x_range"):
        script_lines.append("ax.set_xlim(left={}, right={})\n".format(*plot_handler.x_range))
