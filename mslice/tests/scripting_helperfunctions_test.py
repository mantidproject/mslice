import unittest
from unittest import mock
from mslice.scripting.helperfunctions import header, add_header, add_plot_statements, add_slice_plot_statements, \
    COMMON_PACKAGES, MPL_COLORS_IMPORT, NUMPY_IMPORT, add_overplot_statements, add_cut_plot_statements, add_cut_lines, \
    add_cut_lines_with_width, add_plot_options, hide_lines
from mslice.plotting.plot_window.cut_plot import CutPlot
from mslice.plotting.plot_window.slice_plot import SlicePlot
from mslice.cli.helperfunctions import _function_to_intensity
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from mslice.models.cut.cut import Cut
from mslice.models.axis import Axis


class ScriptingHelperFunctionsTest(unittest.TestCase):

    def assign_slice_parameters(self, plot_handler, intensity=True, temp_dependent=True):
        plot_handler.colorbar_label = 'colorbar_label'
        plot_handler.colorbar_log = True
        plot_handler.colorbar_range = (0, 30)
        plot_handler.temp_dependent = temp_dependent
        plot_handler.temp = 30
        plot_handler.intensity = intensity
        plot_handler.intensity_method = 'show_scattering_function'
        plot_handler.y_range = (0, 10)
        plot_handler.x_range = (0, 10)

    def assign_cut_parameters(self, plot_handler):
        plot_handler.title = 'Title'
        plot_handler.y_label = 'y_label'
        plot_handler.x_label = 'x_label'
        plot_handler.y_grid = 'y_grid'
        plot_handler.x_grid = 'x_grid'
        plot_handler.y_range = (1, 10)
        plot_handler.x_range = (1, 10)

    def test_that_header_works_as_expected_for_cuts(self):
        plot_handler = mock.MagicMock(spec=CutPlot)
        plot_handler.x_log, plot_handler.y_log = True, True

        return_value = header(plot_handler)
        self.assertIn("\n".join(COMMON_PACKAGES), return_value)
        self.assertIn("\n".join(NUMPY_IMPORT), return_value)

    def test_that_header_works_as_expected_for_slices(self):
        plot_handler = mock.MagicMock(spec=SlicePlot)
        plot_handler.colorbar_log = True

        return_value = header(plot_handler)
        self.assertIn("\n".join(COMMON_PACKAGES), return_value)
        self.assertIn("\n".join(MPL_COLORS_IMPORT), return_value)

    def test_that_add_header_works_as_expected(self):
        plot_handler = mock.MagicMock(spec=SlicePlot)
        plot_handler.colorbar_log = True
        script_lines = []

        add_header(script_lines, plot_handler)
        self.assertIn("\n".join(COMMON_PACKAGES), script_lines)
        self.assertIn("\n".join(MPL_COLORS_IMPORT), script_lines)

    @mock.patch('mslice.scripting.helperfunctions.add_header')
    @mock.patch('mslice.scripting.helperfunctions.add_cut_plot_statements')
    @mock.patch('mslice.scripting.helperfunctions.add_overplot_statements')
    def test_that_add_plot_statements_works_as_expected_for_cuts(self, add_overplot, add_cut, add_header):
        plot_handler = mock.MagicMock(spec=CutPlot)
        script_lines = []

        fig = mock.MagicMock()
        ax = fig.add_subplot(111, projection='mslice')

        add_plot_statements(script_lines, plot_handler, ax)

        add_header.assert_called_once_with(script_lines, plot_handler)
        add_cut.assert_called_once_with(script_lines, plot_handler, ax)
        add_overplot.assert_called_once_with(script_lines, plot_handler)

        self.assertIn("mc.Show()\n", script_lines)
        self.assertIn('fig = plt.gcf()\n', script_lines)
        self.assertIn('fig.clf()\n', script_lines)
        self.assertIn('ax = fig.add_subplot(111, projection="mslice")\n', script_lines)

    @mock.patch('mslice.scripting.helperfunctions.add_header')
    @mock.patch('mslice.scripting.helperfunctions.add_slice_plot_statements')
    @mock.patch('mslice.scripting.helperfunctions.add_overplot_statements')
    def test_that_add_plot_statements_works_as_expected_for_slices(self, add_overplot, add_slice, add_header):
        plot_handler = mock.MagicMock(spec=SlicePlot)
        script_lines = []

        fig = mock.MagicMock()
        ax = fig.add_subplot(111, projection='mslice')

        add_plot_statements(script_lines, plot_handler, ax)

        add_header.assert_called_once_with(script_lines, plot_handler)
        add_slice.assert_called_once_with(script_lines, plot_handler)
        add_overplot.assert_called_once_with(script_lines, plot_handler)

        self.assertIn("mc.Show()\n", script_lines)
        self.assertIn('fig = plt.gcf()\n', script_lines)
        self.assertIn('fig.clf()\n', script_lines)
        self.assertIn('ax = fig.add_subplot(111, projection="mslice")\n', script_lines)

    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    @mock.patch('mslice.scripting.helperfunctions.add_plot_options')
    def test_that_add_slice_plot_statements_works_as_expected_with_intensity_and_temp_dependence(self, add_plot, gfm):
        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(SlicePlot)
        self.assign_slice_parameters(plot_handler, intensity=True, temp_dependent=True)
        script_lines = []

        add_slice_plot_statements(script_lines, plot_handler)

        cache = plot_handler._slice_plotter_presenter._slice_cache
        intensity = _function_to_intensity[plot_handler.intensity_method]
        slice = cache[plot_handler.ws_name]
        momentum_axis = str(slice.momentum_axis)
        energy_axis = str(slice.energy_axis)
        norm = slice.norm_to_one

        self.assertIn('slice_ws = mc.Slice(ws_{}, Axis1="{}", Axis2="{}", NormToOne={})\n\n'.format(
            plot_handler.ws_name.replace(".", "_"), momentum_axis, energy_axis, norm), script_lines)
        self.assertIn('mesh = ax.pcolormesh(slice_ws, cmap="{}", intensity="{}", temperature={})\n'.format(
            cache[plot_handler.ws_name].colourmap, intensity, plot_handler.temp), script_lines)
        self.assertIn("cb = plt.colorbar(mesh, ax=ax)\n", script_lines)
        self.assertIn("cb.set_label('{}', labelpad=20, rotation=270, picker=5)\n".format(plot_handler.colorbar_label),
                      script_lines)
        self.assertIn("mesh.set_norm(colors.LogNorm({}, {}))\n".format(0.001, 30), script_lines)
        add_plot.assert_called_once_with(script_lines, plot_handler)

    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    def test_that_add_slice_plot_statements_works_as_expected_without_intensity(self, gfm):
        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(SlicePlot)
        self.assign_slice_parameters(plot_handler, intensity=False)
        script_lines = []

        add_slice_plot_statements(script_lines, plot_handler)
        cache = plot_handler._slice_plotter_presenter._slice_cache

        self.assertIn('mesh = ax.pcolormesh(slice_ws, cmap="{}")\n'.format(cache[plot_handler.ws_name].colourmap),
                      script_lines)

    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    def test_that_add_slice_plot_statements_works_with_intensity_and_no_temp_dependence(self, gfm):
        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(SlicePlot)
        self.assign_slice_parameters(plot_handler, intensity=True, temp_dependent=False)
        script_lines = []

        add_slice_plot_statements(script_lines, plot_handler)

        cache = plot_handler._slice_plotter_presenter._slice_cache
        intensity = _function_to_intensity[plot_handler.intensity_method]

        self.assertIn('mesh = ax.pcolormesh(slice_ws, cmap="{}", intensity="{}")\n'.format(
            cache[plot_handler.ws_name].colourmap, intensity), script_lines)

    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    def test_that_add_overplot_statements_works_as_expected_with_recoil_element(self, gfm):
        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(SlicePlot)
        self.assign_slice_parameters(plot_handler)
        plot_handler._canvas.figure.gca().lines = [Line2D([1, 2], [1, 2], label="Hydrogen")]
        workspace_name = plot_handler.ws_name
        script_lines = []

        add_overplot_statements(script_lines, plot_handler)

        self.assertIn("ax.recoil(workspace='{}', element='{}')\n".format(workspace_name, "Hydrogen"), script_lines)

    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    def test_that_add_overplot_statements_works_as_expected_with_arbitrary_nuclei(self, gfm):
        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(SlicePlot)
        self.assign_slice_parameters(plot_handler)
        plot_handler._canvas.figure.gca().lines = [Line2D([1, 2], [1, 2], label="Relative Mass 55")]
        workspace_name = plot_handler.ws_name
        script_lines = []

        add_overplot_statements(script_lines, plot_handler)

        self.assertIn("ax.recoil(workspace='{}', rmm={})\n".format(workspace_name, 55), script_lines)

    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    def test_that_add_overplot_statements_works_as_expected_with_bragg_peaks_elements(self, gfm):
        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(SlicePlot)
        plot_handler._canvas.figure.gca().lines = [Line2D([1, 2], [1, 2], label="Tantalum")]
        workspace_name = plot_handler.ws_name
        script_lines = []

        add_overplot_statements(script_lines, plot_handler)

        self.assertIn("ax.bragg(workspace='{}', element='{}')\n".format(workspace_name, "Tantalum"), script_lines)

    @mock.patch('mslice.scripting.helperfunctions.add_plot_options')
    @mock.patch('mslice.scripting.helperfunctions.add_cut_lines')
    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    def test_that_add_cut_plot_statements_works_as_expected(self, gfm, add_cut_lines, add_plot_options):
        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(CutPlot)
        script_lines = []

        fig = mock.MagicMock()
        ax = fig.add_subplot(111, projection='mslice')

        add_cut_plot_statements(script_lines, plot_handler, ax)

        add_cut_lines.assert_called_once_with(script_lines, plot_handler, ax)
        add_plot_options.assert_called_once_with(script_lines, plot_handler)

        self.assertIn("ax.set_xscale('symlog', linthreshx=pow(10, np.floor(np.log10({}))))\n".format(
            plot_handler.x_axis_min), script_lines)

        self.assertIn("ax.set_yscale('symlog', linthreshy=pow(10, np.floor(np.log10({}))))\n".format(
            plot_handler.y_axis_min), script_lines)

    @mock.patch('mslice.scripting.helperfunctions.add_cut_lines_with_width')
    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    def test_that_add_cut_lines_works_as_expected(self, gfm, add_cut_lines_with_width):
        fig = mock.MagicMock()
        ax = fig.add_subplot(111, projection='mslice')

        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(CutPlot)

        script_lines = []

        add_cut_lines(script_lines, plot_handler, ax)

        cuts = plot_handler._cut_plotter_presenter._cut_cache_dict[ax]
        errorbars = plot_handler._canvas.figure.gca().containers

        add_cut_lines_with_width.assert_called_once_with(errorbars, script_lines, cuts)

    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    def test_that_add_plot_options_works_as_expected(self, gfm):
        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(CutPlot)
        self.assign_cut_parameters(plot_handler)
        script_lines = []

        add_plot_options(script_lines, plot_handler)

        self.assertIn("ax.set_title('{}')\n".format(plot_handler.title), script_lines)
        self.assertIn("ax.set_ylabel(r'{}')\n".format(plot_handler.y_label), script_lines)
        self.assertIn("ax.set_xlabel(r'{}')\n".format(plot_handler.x_label), script_lines)
        self.assertIn("ax.grid({}, axis='y')\n".format(plot_handler.y_grid), script_lines)
        self.assertIn("ax.grid({}, axis='x')\n".format(plot_handler.x_grid), script_lines)
        self.assertIn("ax.set_ylim(bottom={}, top={})\n".format(*plot_handler.y_range), script_lines)
        self.assertIn("ax.set_xlim(left={}, right={})\n".format(*plot_handler.x_range), script_lines)

    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    def test_that_add_plot_options_works_as_expected_when_plots_options_are_not_changed(self, gfm):
        plot_handler = gfm.get_active_figure().plot_handler
        plot_handler.add_mock_spec(CutPlot)
        plot_handler.is_changed.return_value = False
        self.assign_cut_parameters(plot_handler)
        script_lines = []

        add_plot_options(script_lines, plot_handler)

        self.assertIn("ax.set_title('{}')\n".format(plot_handler.title), script_lines)

    def test_that_add_cut_lines_with_width_works_as_expected_without_intensity_range(self):
        x_data, y_data = np.arange(0, 10), np.arange(0, 10)
        plt.errorbar(x_data, y_data, linestyle='-', linewidth=1.5, color='blue', label='errorbar_label')
        errorbars = plt.gca().containers
        cut = Cut(Axis('|Q|', '1', '3', '1'), Axis('DelataE', '-1', '1', '0'), None, None, True, '2')
        cut.workspace_name = 'ws_name'

        cuts = [cut]
        script_lines = []

        add_cut_lines_with_width(errorbars, script_lines, cuts)

        self.assertIn(
            'cut_ws_{} = mc.Cut(ws_{}, CutAxis="{}", IntegrationAxis="{}", NormToOne={}, Algorithm={})\n'.format(
                0, 'ws_name', cuts[0].cut_axis, cuts[0].integration_axis, cuts[0].norm_to_one, "\"Integration\""),
            script_lines)

        self.assertIn(
            'ax.errorbar(cut_ws_{}, label="{}", color="{}", marker="{}", ls="{}", lw={})\n\n'.format(
                0, 'errorbar_label', 'blue', None, '-', 1.5), script_lines)

    def test_that_add_cut_lines_with_width_works_as_expected_with_intensity_range(self):
        x_data, y_data = np.arange(0, 10), np.arange(0, 10)
        plt.errorbar(x_data, y_data, linestyle='-', linewidth=1.5, color='blue', label='errorbar_label')
        errorbars = plt.gca().containers
        cut = Cut(Axis('|Q|', '1', '3', '1'), Axis('DelataE', '-1', '1', '0'), 1.0, 2.0, True, '2')
        cut.workspace_name = 'ws_name'

        cuts = [cut]
        script_lines = []

        add_cut_lines_with_width(errorbars, script_lines, cuts)

        self.assertIn(
            'ax.errorbar(cut_ws_{}, label="{}", color="{}", marker="{}", ls="{}", lw={}, '
            'intensity_range={})\n\n'.format(0, 'errorbar_label', 'blue', None, '-', 1.5, (1.0, 2.0)),
            script_lines)

    def test_that_add_cut_lines_with_width_works_as_expected_with_multiple_cuts(self):
        x_data, y_data = np.arange(0, 10), np.arange(0, 10)
        for i, color in enumerate(['red', 'blue', 'green']):
            plt.errorbar(x_data, y_data, linestyle='-', linewidth=1.5, color='blue', label='error_label_{}'.format(i))
        errorbars = plt.gca().containers
        cut_0 = Cut(Axis('|Q|', '1', '3', '1'), Axis('DeltaE', '-1', '1', '0'), 1.0, 2.0, True, '1')
        cut_0.workspace_name = 'ws_0'

        cut_2 = Cut(Axis('|Q|', '1', '3', '1'), Axis('DeltaE', '-1', '1', '0'), 1.0, 2.0, True, '2')
        cut_2.workspace_name = 'ws_1'

        cuts = [cut_0, cut_2]
        script_lines = []

        add_cut_lines_with_width(errorbars, script_lines, cuts)

        self.assertIn(
            'cut_ws_{} = mc.Cut(ws_{}, CutAxis="{}", IntegrationAxis="{}", NormToOne={}, Algorithm={})\n'.format(
                0, 'ws_0', cuts[0].cut_axis, "DeltaE,-1.0,0.0,0.0", cuts[0].norm_to_one, "\"Integration\""), script_lines)

        self.assertIn(
            'cut_ws_{} = mc.Cut(ws_{}, CutAxis="{}", IntegrationAxis="{}", NormToOne={}, Algorithm={})\n'.format(
                1, 'ws_0', cuts[0].cut_axis, "DeltaE,0.0,1.0,0.0", cuts[0].norm_to_one, "\"Integration\""), script_lines)

        self.assertIn(
            'cut_ws_{} = mc.Cut(ws_{}, CutAxis="{}", IntegrationAxis="{}", NormToOne={}, Algorithm={})\n'.format(
                2, 'ws_1', cuts[1].cut_axis, cuts[1].integration_axis, cuts[1].norm_to_one, "\"Integration\""),
            script_lines)

        # Each mc.Cut statement has a corresponding errorbar statement
        self.assertEqual(len(script_lines), 6)

    def test_show_or_hide_containers_in_script(self):
        fig, ax = plt.subplots()
        ax.errorbar([1], [2], [0.3], label="label1")
        ax.errorbar([2], [4], [0.6], label="label2")
        plot_handler = mock.MagicMock(spec=CutPlot)
        plot_handler.get_line_visible = mock.MagicMock(side_effect=[False, True])
        script_lines = []
        hide_lines(script_lines, plot_handler, ax)
        self.assertIn("from mslice.cli.helperfunctions import hide_a_line_and_errorbars,"
                      " append_visible_handle_and_label\n", script_lines)
        self.assertIn("from mslice.util.compat import legend_set_draggable\n\n", script_lines)

        self.assertIn("# hide lines, errorbars, and legends\n", script_lines)
        self.assertIn("handles, labels = ax.get_legend_handles_labels()\n", script_lines)
        self.assertIn("visible_handles = []\n", script_lines)
        self.assertIn("visible_labels = []\n", script_lines)

        self.assertIn("\nhide_a_line_and_errorbars(ax, 0)\n", script_lines)

        self.assertNotIn("\nhide_a_line_and_errorbars(ax, 1)\n", script_lines)
        self.assertIn("\nappend_visible_handle_and_label(visible_handles, handles, visible_labels, labels, 1)\n",
                      script_lines)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
