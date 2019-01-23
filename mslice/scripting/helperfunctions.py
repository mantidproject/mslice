from matplotlib.lines import Line2D
from datetime import datetime

PACKAGES = {'mslice.cli': 'mc', 'mslice.plotting.pyplot': 'plt'}


def cleanup(script_lines):
    for line in script_lines[-1:]:
        if line.startswith('Load'):
            index = script_lines.index(line)
            return script_lines[index:]


def header():
    statements = ['# Python Script Generated by Mslice on {}\n'.format(datetime.now())]

    for package in PACKAGES:
        statements.append('import {} as {}\n'.format(package, PACKAGES[package]))

    return statements


def add_header(script_lines):
    for i, statement in enumerate(header()):
        script_lines.insert(i, statement)


def add_plot_statements(script_lines, plot_handler):
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    from mslice.plotting.plot_window.cut_plot import CutPlot

    add_header(script_lines)

    line_no = len(script_lines)
    script_lines.insert(line_no - 1, '\n')
    script_lines[line_no] = 'ws = mc.{}\n'.format(script_lines[line_no])

    if plot_handler is not None:
        if isinstance(plot_handler, SlicePlot):
            script_lines.append('slice_ws = mc.Slice(ws)\n\n')
            script_lines.append('fig = plt.gcf()\n')
            script_lines.append('ax = fig.add_subplot(111, projection=\'mslice\')\n')
            script_lines.append('mesh = ax.pcolormesh(slice_ws, cmap=\'viridis\')\n')
            script_lines.append('cb = plt.colorbar(mesh, ax=ax)\n')
            script_lines.append('cb.set_label(\'{}\', labelpad=20, rotation=270, picker=5)\n'.format(
                plot_handler.colorbar_label))

            add_intensity_statements(script_lines, plot_handler)
            add_slice_plot_statements(script_lines, plot_handler)
            add_overplot_statements(script_lines, plot_handler)
        elif isinstance(plot_handler, CutPlot):
            add_cut_plot_statements(script_lines, plot_handler)

        script_lines.append('mc.Show()\n')

    return script_lines


def add_slice_plot_statements(script_lines, plot_handler):
    script_lines.append('ax.set_title(\'{}\')\n'.format(plot_handler.title)
                        if plot_handler.is_changed('title') else '')

    script_lines.append('ax.set_ylabel(\'{}\')\n'.format(plot_handler.y_label)
                        if plot_handler.is_changed('y_label') else '')
    script_lines.append('ax.set_xlabel(\'{}\')\n'.format(plot_handler.x_label)
                        if plot_handler.is_changed('x_label') else '')

    script_lines.append('ax.grid({}, axis=\'y\')\n'.format(plot_handler.y_grid)
                        if plot_handler.is_changed('y_grid') else '')
    script_lines.append('ax.grid({}, axis=\'x\')\n'.format(plot_handler.x_grid)
                        if plot_handler.is_changed('x_grid') else '')

    script_lines.append('ax.set_ylim(bottom={}, top={})\n'.format(*plot_handler.y_range)
                        if plot_handler.is_changed('y_range') else '')
    script_lines.append('ax.set_xlim(left={}, right={})\n'.format(*plot_handler.x_range)
                        if plot_handler.is_changed('x_range') else '')

    script_lines.append(
        'ax.change_axis_scale(colorbar_range={}, logarithmic={})\n'.format(
            plot_handler.colorbar_range, plot_handler.colorbar_log)
        if plot_handler.is_changed('colorbar_range') or plot_handler.is_changed('colorbar_log') else '')


def add_overplot_statements(script_lines, plot_handler):
    ax = plot_handler._canvas.figure.gca()
    line_artists = [artist for artist in ax.get_children() if isinstance(artist, Line2D)]

    for line in line_artists:
        label = line._label
        key = 1 if label == 'Hydrogen' else 2 if label == 'Deuterium' else 4 if label == 'Helium' else label
        key = int(label.split()[-1]) if "Relative" in label else key
        recoil = isinstance(key, int)  # Recoil line keys are integers

        if recoil:
            script_lines.append(
                'ax.recoil_line(workspace_name=\'{}\', key={}, recoil={}, cif={})\n'.format(
                    plot_handler.ws_name, key, recoil, None))  # Does not yet account for CIF files
        else:
            script_lines.append(
                'ax.recoil_line(workspace_name=\'{}\', key=\'{}\', recoil={}, cif={})\n'.format(
                    plot_handler.ws_name, key, recoil, None))


def add_intensity_statements(script_lines, plot_handler):
    if plot_handler.default_options['intensity'] is True:
        script_lines.append(
            'ax = ax.intensity_plot(workspace_name=\'{}\', method_name=\'{}\', '
            'temp_value={}, temp_dependent={}, label={})\n'.format(
                plot_handler.ws_name, plot_handler.default_options['intensity_method'],
                plot_handler.default_options['temp'], plot_handler.default_options['temp_dependent'],
                plot_handler.colorbar_label))


def add_cut_plot_statements(script_lines, plot_handler):
    script_lines.append('cut_ws = mc.Cut(ws)')
    script_lines.append('ax.plot(cut_ws)')

    script_lines.append('ax.set_title(\'{}\')\n'.format(plot_handler.title))

    script_lines.append('ax.set_ylabel(\'{}\')\n'.format(plot_handler.y_label))
    script_lines.append('ax.set_xlabel(\'{}\')\n'.format(plot_handler.x_label))

    script_lines.append('ax.grid({}, axis=\'y\')\n'.format(plot_handler.y_grid))
    script_lines.append('ax.grid({}, axis=\'x\')\n'.format(plot_handler.x_grid))

    script_lines.append('ax.set_ylim(left={}, right={})\n'.format(*plot_handler.y_range))
    script_lines.append('ax.set_xlim(left={}, right={})\n'.format(*plot_handler.x_range))
