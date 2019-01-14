import matplotlib.colors as colors
from matplotlib.lines import Line2D

PACKAGES = {'mslice.cli': 'mc', 'matplotlib.pyplot': 'plt', 'mslice.scripting.helperfunctions': 'scripting',
            'mslice.app.presenters': 'presenters'}


def import_statements():
    statements = []
    for package in PACKAGES:
        statements.append('import {} as {}\n'.format(package, PACKAGES[package]))
    statements.append("fig, ax = plt.subplots(subplot_kw={'projection': 'mslice'})\n\n")
    return statements


def add_import_statements(filename):
    with open(filename, 'r+') as generated_script:
        script_lines = generated_script.readlines()
        for i, statement in enumerate(import_statements()):
            script_lines.insert(3 + i, statement)  # Used 3 since the first 3 lines of generated scripts are comments
        generated_script.seek(0)
        generated_script.writelines(script_lines)


def add_plot_statements(filename, plot_handler):
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    from mslice.plotting.plot_window.cut_plot import CutPlot

    with open(filename, 'r+') as generated_script:
        script_lines = generated_script.readlines()
        line_no = len(script_lines)
        script_lines.insert(line_no - 1, '\n')
        script_lines[line_no] = 'ws = {}'.format(script_lines[line_no])

        if plot_handler is not None:
            if isinstance(plot_handler, SlicePlot):
                add_slice_plot_statements(script_lines, plot_handler)
                add_overplot_statements(script_lines, plot_handler)
                script_lines.append('\nmc.Show()\n')
            elif isinstance(plot_handler, CutPlot):
                add_cut_plot_statements(script_lines, plot_handler)
                script_lines.append('\nmc.Show()\n')

        generated_script.seek(0)
        generated_script.writelines(script_lines)


def add_slice_plot_statements(script_lines, plot_handler):
    script_lines.append('slice_ws = mc.Slice(ws)\n')
    script_lines.append('colormesh = ax.pcolormesh(slice_ws)\n\n')

    script_lines.append('#User Changes\n')  # Could put checks in slice_plot to only write what has changed

    script_lines.append('ax.set_title(\'{}\')\n'.format(plot_handler.title))

    script_lines.append('ax.set_ylabel(\'{}\')\n'.format(plot_handler.y_label))
    script_lines.append('ax.set_xlabel(\'{}\')\n'.format(plot_handler.x_label))

    script_lines.append('ax.grid({}, axis=\'y\')\n'.format(plot_handler.y_grid))
    script_lines.append('ax.grid({}, axis=\'x\')\n'.format(plot_handler.x_grid))

    script_lines.append('ax.set_ylim(left={}, right={})\n'.format(*plot_handler.y_range))
    script_lines.append('ax.set_xlim(left={}, right={})\n'.format(*plot_handler.x_range))

    script_lines.append('colormesh.set_clim({})\n'.format(plot_handler.colorbar_range))
    script_lines.append('colormesh.set_label(\'{}\')\n'.format(plot_handler.colorbar_label))
    script_lines.append('scripting.change_axis_scale(ax, fig, {}, {})\n\n'.format(plot_handler.colorbar_range,
                                                                                  plot_handler.colorbar_log))


def add_overplot_statements(script_lines, plot_handler):
    ax = plot_handler._canvas.figure.gca()
    line_artists = [artist for artist in ax.get_children() if isinstance(artist, Line2D)]

    for line in line_artists:
        label = line._label
        key = 1 if label == 'Hydrogen' else 2 if label == 'Deuterium' else 4 if label == 'Helium' else label
        recoil = isinstance(key, int)  # Recoil line keys are integers

        if recoil:
            script_lines.append(
                'presenters.get_slice_plotter_presenter().add_overplot_line(\'{}\', {}, {}, {})\n'.format(
                    plot_handler.ws_name, key, recoil, None))  # Does not yet account for CIF files
        else:
            script_lines.append(
                'presenters.get_slice_plotter_presenter().add_overplot_line(\'{}\', \'{}\', {}, {})\n'.format(
                    plot_handler.ws_name, key, recoil, None))


def add_cut_plot_statements(script_lines, plot_handler):
    script_lines.append('cut_ws = mc.Cut(ws)')
    script_lines.append('ax.plot(cut_ws)')
    return plot_handler


def change_axis_scale(ax, fig, colorbar_range, logarithmic):
    colormesh = ax.collections[0]
    vmin, vmax = colorbar_range
    if logarithmic:
        colormesh.colorbar.remove()
        if vmin <= float(0):
            vmin = 0.001
        colormesh.set_clim((vmin, vmax))
        norm = colors.LogNorm(vmin, vmax)
        colormesh.set_norm(norm)
        fig.colorbar(colormesh)
    else:
        colormesh.colorbar.remove()
        colormesh.set_clim((vmin, vmax))
        norm = colors.Normalize(vmin, vmax)
        colormesh.set_norm(norm)
        fig.colorbar(colormesh)
