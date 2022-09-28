from datetime import datetime
from distutils.version import LooseVersion
from mslice.cli.helperfunctions import _function_to_intensity
from mslice.models.labels import get_recoil_key
from matplotlib import __version__ as mpl_version
import re

COMMON_PACKAGES = ["import mslice.cli as mc", "import mslice.plotting.pyplot as plt\n\n"]
MPL_COLORS_IMPORT = ["\nimport matplotlib.colors as colors\n"]
NUMPY_IMPORT = ["\nimport numpy as np\n"]
LOG_SCALE_MIN = 0.001


def header(plot_handler):
    """Creates a list of import statements to be used in the generated script header"""
    from mslice.plotting.plot_window.cut_plot import CutPlot
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    statements = ["# Python Script Generated by Mslice on {}\n".format(datetime.now().replace(microsecond=0))]

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
    script_lines.append("\n")


def add_plot_statements(script_lines, plot_handler, ax):
    """Adds plot statements to the script lines used to generate the python script"""
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    from mslice.plotting.plot_window.cut_plot import CutPlot

    add_header(script_lines, plot_handler)

    script_lines.append('fig = plt.gcf()\n')
    script_lines.append('fig.clf()\n')
    script_lines.append('ax = fig.add_subplot(111, projection="mslice")\n')

    if plot_handler is not None:
        if isinstance(plot_handler, SlicePlot):
            add_slice_plot_statements(script_lines, plot_handler)
            add_overplot_statements(script_lines, plot_handler)
        elif isinstance(plot_handler, CutPlot):
            add_cut_plot_statements(script_lines, plot_handler, ax)
            add_overplot_statements(script_lines, plot_handler)

        script_lines.append("mc.Show()\n")

    return script_lines


def add_slice_plot_statements(script_lines, plot_handler):
    cache = plot_handler._slice_plotter_presenter._slice_cache

    slice = cache[plot_handler.ws_name]
    momentum_axis = str(slice.momentum_axis)
    energy_axis = str(slice.energy_axis)
    norm = slice.norm_to_one

    script_lines.append('slice_ws = mc.Slice(ws_{}, Axis1="{}", Axis2="{}", NormToOne={})\n\n'.format(
        plot_handler.ws_name.replace(".", "_"), momentum_axis, energy_axis, norm))

    if plot_handler.intensity is True:
        intensity = _function_to_intensity[plot_handler.intensity_method]
        if plot_handler.temp_dependent:
            script_lines.append('mesh = ax.pcolormesh(slice_ws, cmap="{}", intensity="{}", temperature={})\n'.format(
                cache[plot_handler.ws_name].colourmap, intensity, plot_handler.temp))
        else:
            script_lines.append('mesh = ax.pcolormesh(slice_ws, cmap="{}", intensity="{}")\n'.format(
                cache[plot_handler.ws_name].colourmap, intensity))
    else:
        script_lines.append('mesh = ax.pcolormesh(slice_ws, cmap="{}")\n'.format(cache[plot_handler.ws_name].colourmap))

    script_lines.append("mesh.set_clim({}, {})\n".format(*plot_handler.colorbar_range))
    if plot_handler.colorbar_log:
        min, maximum = plot_handler.colorbar_range[0], plot_handler.colorbar_range[1]
        min = max(min, LOG_SCALE_MIN)
        script_lines.append("mesh.set_norm(colors.LogNorm({}, {}))\n".format(min, maximum))

    script_lines.append("cb = plt.colorbar(mesh, ax=ax)\n")
    script_lines.append(f"cb.set_label('{plot_handler.colorbar_label}', labelpad=20, rotation=270, picker=5, "
                        f"fontsize={plot_handler.colorbar_label_size})\n")
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
                script_lines.append("ax.recoil(workspace='{}', rmm={})\n".format(plot_handler.ws_name, rmm))
            else:
                script_lines.append("ax.recoil(workspace='{}', element='{}')\n".format(plot_handler.ws_name, element))
        else:
            if cif is None:
                script_lines.append("ax.bragg(workspace='{}', element='{}')\n".format(plot_handler.ws_name, element))

            else:
                script_lines.append("ax.bragg(workspace='{}', cif='{}')\n".format(plot_handler.ws_name, cif))


def add_cut_plot_statements(script_lines, plot_handler, ax):
    """Adds cut specific statements to the script"""

    add_cut_lines(script_lines, plot_handler, ax)
    add_plot_options(script_lines, plot_handler)

    if plot_handler.is_changed("x_log"):
        x_axis_str = "x" if LooseVersion(mpl_version) < LooseVersion('3.3') else ""
        script_lines.append(f"ax.set_xscale('symlog', "
                            f"linthresh{x_axis_str}=pow(10, np.floor(np.log10({plot_handler.x_axis_min}))))\n")

    if plot_handler.is_changed("y_log"):
        y_axis_str = "y" if LooseVersion(mpl_version) < LooseVersion('3.3') else ""
        script_lines.append(f"ax.set_yscale('symlog', "
                            f"linthresh{y_axis_str}=pow(10, np.floor(np.log10({plot_handler.y_axis_min}))))\n")

def add_cut_lines(script_lines, plot_handler, ax):
    cuts = plot_handler._cut_plotter_presenter._cut_cache_dict[ax]
    errorbars = plot_handler._canvas.figure.gca().containers
    itensity_correction = plot_handler.intensity_method
    itensity_correction = itensity_correction[5:] if itensity_correction else itensity_correction
    add_cut_lines_with_width(errorbars, script_lines, cuts, itensity_correction)
    hide_lines(script_lines, plot_handler, ax)


def hide_lines(script_lines, plot_handler, ax):
    """ Check if the line needs to be shown or not (hidden).
    If the line is hidden, corresponding errorbars and legend
    are also hidden.
    """
    script_lines.append("from mslice.cli.helperfunctions import hide_a_line_and_errorbars,"
                        " append_visible_handle_and_label\n")
    script_lines.append("from mslice.util.compat import legend_set_draggable\n\n")

    script_lines.append("# hide lines, errorbars, and legends\n")
    script_lines.append("handles, labels = ax.get_legend_handles_labels()\n")
    script_lines.append("visible_handles = []\n")
    script_lines.append("visible_labels = []\n")
    idx = -1
    for container in ax.containers:
        idx += 1
        line_visible = plot_handler.get_line_visible(idx)
        if line_visible:
            # only add handles and labels if the corresponding line is shown
            script_lines.append(
                f"\nappend_visible_handle_and_label(visible_handles, handles, visible_labels, labels, {idx:d})\n")
        else:
            script_lines.append(f"\nhide_a_line_and_errorbars(ax, {idx:d})\n")
    script_lines.append("\nlegend_set_draggable(ax.legend(visible_handles, visible_labels,"
                        " fontsize='medium'), True)\n\n")


def add_cut_lines_with_width(errorbars, script_lines, cuts, intensity_correction):
    """Adds the cut statements for each interval of the cuts that were plotted"""
    index = 0  # Required as we run through the loop multiple times for each cut
    for cut in cuts:
        integration_start = cut.integration_axis.start
        integration_end = cut.integration_axis.end
        cut_start, cut_end = integration_start, min(integration_start + cut.width, integration_end)
        intensity_range = (cut.intensity_start, cut.intensity_end)
        norm_to_one = cut.norm_to_one
        algo_str = '' if 'Rebin' in cut.algorithm else ', Algorithm="{}"'.format(cut.algorithm)

        while cut_start != cut_end and index < len(errorbars):
            cut.integration_axis.start = cut_start
            cut.integration_axis.end = cut_end
            cut_axis = str(cut.cut_axis)
            integration_axis = str(cut.integration_axis)

            errorbar = errorbars[index]
            colour = errorbar.lines[0]._color
            marker = errorbar.lines[0]._marker._marker
            style = errorbar.lines[0]._linestyle
            width = errorbar.lines[0]._linewidth
            label = errorbar._label

            intensity_correction_arg = intensity_correction if not intensity_correction else f'"{intensity_correction}"'
            script_lines.append('cut_ws_{} = mc.Cut(ws_{}, CutAxis="{}", IntegrationAxis="{}", '
                                'NormToOne={}{}, IntensityCorrection={}, SampleTemperature={})'
                                '\n'.format(index, replace_ws_special_chars(cut.parent_ws_name), cut_axis, integration_axis,
                                            norm_to_one, algo_str, intensity_correction_arg, cut.raw_sample_temp))

            if intensity_range != (None, None):
                script_lines.append(
                    'ax.errorbar(cut_ws_{}, label="{}", color="{}", marker="{}", ls="{}", '
                    'lw={}, intensity_range={})\n\n'.format(index, label, colour, marker, style, width,
                                                            intensity_range))
            else:
                script_lines.append(
                    'ax.errorbar(cut_ws_{}, label="{}", color="{}", marker="{}", ls="{}", '
                    'lw={})\n\n'.format(index, label, colour, marker, style, width))

            cut_start, cut_end = cut_end, min(cut_end + cut.width, integration_end)
            index += 1
        cut.reset_integration_axis(cut.start, cut.end)


def add_plot_options(script_lines, plot_handler):
    """Adds lines that change the plot options if they were modified"""
    script_lines.append(f"ax.set_title('{plot_handler.title}', fontsize={plot_handler.title_size})\n")

    if plot_handler.is_changed("y_label") or plot_handler.is_changed("y_label_size"):
        script_lines.append(f"ax.set_ylabel(r'{plot_handler.y_label}', fontsize={plot_handler.y_label_size})\n")

    if plot_handler.is_changed("x_label") or plot_handler.is_changed("x_label_size"):
        script_lines.append(f"ax.set_xlabel(r'{plot_handler.x_label}', fontsize={plot_handler.x_label_size})\n")

    if plot_handler.is_changed("y_grid"):
        script_lines.append("ax.grid({}, axis='y')\n".format(plot_handler.y_grid))

    if plot_handler.is_changed("x_grid"):
        script_lines.append("ax.grid({}, axis='x')\n".format(plot_handler.x_grid))

    if plot_handler.is_changed("y_range"):
        script_lines.append("ax.set_ylim(bottom={}, top={})\n".format(*plot_handler.y_range))

    if plot_handler.is_changed("x_range"):
        script_lines.append("ax.set_xlim(left={}, right={})\n".format(*plot_handler.x_range))

    if plot_handler.is_changed("y_range_font_size"):
        script_lines.append(f"ax.yaxis.set_tick_params(labelsize={plot_handler.y_range_font_size})\n")

    if plot_handler.is_changed("x_range_font_size"):
        script_lines.append(f"ax.xaxis.set_tick_params(labelsize={plot_handler.x_range_font_size})\n")

    from mslice.plotting.plot_window.cut_plot import CutPlot
    if isinstance(plot_handler, CutPlot) and plot_handler.is_changed("waterfall"):
        script_lines.append("ax.set_waterfall({}, x_offset={}, y_offset={})\n".format(plot_handler.waterfall,
                                                                                      plot_handler.waterfall_x,
                                                                                      plot_handler.waterfall_y))


def replace_ws_special_chars(workspace_name):
    rep = {".": "_", "(": "_", ")": "_", ",": "_"}
    pattern = re.compile("|".join([re.escape(key) for key in rep.keys()]))
    new_ws_name = pattern.sub(lambda m: rep[m.group(0)], workspace_name)
    while new_ws_name[len(new_ws_name)-1:] == '_':
        new_ws_name = new_ws_name[:len(new_ws_name) - 1]
    return new_ws_name
