import unittest
import mock
import numpy as np
from mslice.workspace import wrap_workspace
from mantid.simpleapi import AddSampleLog, CreateSampleWorkspace
from mslice.plotting.plot_window.cut_plot import CutPlot
from mslice.models.cut.cut import Cut
from mslice.models.axis import Axis
from mslice.models.workspacemanager.workspace_provider import add_workspace
from mslice.scripting import (
    preprocess_lines,
    generate_script_lines,
    get_algorithm_kwargs,
    generate_script,
)
from tests.testhelpers.fake_objects import FakeClipboard, FakeFile


class ScriptingTest(unittest.TestCase):
    def create_workspace(self, name):
        workspace = CreateSampleWorkspace(
            OutputWorkspace=name,
            NumBanks=1,
            BankPixelWidth=5,
            XMin=0.1,
            XMax=3.1,
            BinWidth=0.1,
            XUnit="DeltaE",
        )
        AddSampleLog(Workspace=workspace, LogName="Ei", LogText="3.", LogType="Number")
        sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30, 25).transpose()
        for i in range(workspace.getNumberHistograms()):
            workspace.setY(i, sim_scattering_data[i])
        workspace = wrap_workspace(workspace, name)
        workspace.is_PSD = False
        workspace.limits["MomentumTransfer"] = [0.1, 3.1, 0.1]
        workspace.limits["|Q|"] = [0.1, 3.1, 0.1]
        workspace.limits["DeltaE"] = [-10, 15, 1]
        workspace.e_fixed = 10
        return workspace

    @mock.patch("builtins.open")
    @mock.patch("mslice.scripting.preprocess_lines")
    @mock.patch("mslice.scripting.add_plot_statements")
    def test_that_generate_script_works_as_expected(
        self, add_plot_statements, preprocess_lines, mock_open
    ):
        workspace = self.create_workspace("test")
        filename = "filename"
        plot_handler = mock.MagicMock()
        plot_window = plot_handler.plot_window

        generate_script(
            workspace.name,
            filename=filename,
            plot_handler=plot_handler,
            window=plot_window,
        )

        add_plot_statements.assert_called_once()
        preprocess_lines.assert_called_once()
        mock_open.assert_called_once()

    @mock.patch("mslice.scripting.get_workspace_handle")
    @mock.patch("mslice.scripting.generate_script_lines")
    @mock.patch("mslice.scripting.get_cut_plotter_presenter")
    def test_that_preprocess_lines_works_as_expected_for_multiple_cuts(
        self, get_cpp, gen_lines, workspace_handle
    ):
        fig = mock.MagicMock()
        ax = fig.add_subplot(111, projection="mslice")

        plot_handler = mock.MagicMock(spec=CutPlot)
        ws_name1 = "ws1"
        ws_name2 = "ws2"
        cut1 = Cut(
            Axis("|Q|", "1", "3", "1"),
            Axis("DelataE", "-1", "1", "0"),
            None,
            None,
            True,
            "2",
        )
        cut1.parent_ws_name = ws_name1
        cut2 = Cut(
            Axis("|Q|", "1", "4", "1"),
            Axis("DelataE", "-1", "1", "0"),
            None,
            None,
            True,
            "2",
        )
        cut2.parent_ws_name = ws_name2
        get_cpp()._cut_cache_dict = {ax: [cut1, cut2]}

        ws1 = mock.Mock()
        ws2 = mock.Mock()

        workspace_handle.side_effect = [ws1, ws2]
        preprocess_lines(ws_name2, plot_handler, ax)

        self.assertIn(mock.call(ws1.raw_ws, ws_name1), gen_lines.call_args_list)
        self.assertIn(mock.call(ws2.raw_ws, ws_name2), gen_lines.call_args_list)
        self.assertEqual(2, gen_lines.call_count)

    @mock.patch("mslice.scripting.get_workspace_handle")
    @mock.patch("mslice.scripting.generate_script_lines")
    def test_that_preprocess_lines_works_as_expected_for_single_plots(
        self, gen_lines, workspace_handle
    ):
        plot_handler = mock.MagicMock()
        ws_name1 = "ws1"
        ws1 = workspace_handle(ws_name1).raw_ws

        fig = mock.MagicMock()
        ax = fig.add_subplot(111, projection="mslice")

        preprocess_lines(ws_name1, plot_handler, ax)

        self.assertIn(mock.call(ws1, ws_name1), gen_lines.call_args_list)
        self.assertEqual(1, gen_lines.call_count)

    @mock.patch("mslice.scripting.get_workspace_handle")
    @mock.patch("mslice.scripting.get_algorithm_kwargs")
    def test_that_generate_script_lines_works_as_expected(
        self, get_alg_kwargs, get_workspace_handle
    ):
        ws_name = self.create_workspace("workspace").name
        raw_ws = get_workspace_handle("workspace").raw_ws

        load_alg = mock.MagicMock()
        load_alg.name.return_value = "Load"
        load_alg_prop_filename = mock.MagicMock()
        load_alg_prop_filename.name.return_value = "Filename"

        make_projection_alg = mock.MagicMock()
        make_projection_alg.name.return_value = "MakeProjection"

        raw_ws.getHistory().getAlgorithmHistories.return_value = [
            load_alg,
            make_projection_alg,
        ]
        get_alg_kwargs.return_value = ("some args", "workspace")

        script_lines = generate_script_lines(raw_ws, ws_name)
        self.assertEqual(get_alg_kwargs.call_count, 2)

        self.assertIn(mock.call(load_alg, []), get_alg_kwargs.call_args_list)
        self.assertIn(mock.call(make_projection_alg, []), get_alg_kwargs.call_args_list)

        make_projection_kwargs, output_ws = get_alg_kwargs(make_projection_alg, ws_name)
        load_kwargs, output_ws = get_alg_kwargs(load_alg, ws_name)

        self.assertIn(
            "ws_{} = mc.{}({})\n".format(ws_name, load_alg.name(), load_kwargs),
            script_lines,
        )
        self.assertIn(
            "ws_{} = mc.{}({})\n".format(
                ws_name, make_projection_alg.name(), make_projection_kwargs
            ),
            script_lines,
        )

    @mock.patch("mslice.scripting.get_workspace_handle")
    @mock.patch("mslice.scripting.get_algorithm_kwargs")
    def test_that_generate_script_lines_works_as_expected_with_only_load(
        self, get_alg_kwargs, get_workspace_handle
    ):
        ws_name = self.create_workspace("workspace").name
        raw_ws = get_workspace_handle("workspace").raw_ws

        load_alg = mock.MagicMock()
        load_alg.name.return_value = "Load"
        some_other_alg = mock.MagicMock()
        some_other_alg.name.return_value = "SomeOtherAlgorithm"
        get_alg_kwargs.return_value = ("some args", "workspace")

        raw_ws.getHistory().getAlgorithmHistories.return_value = [
            some_other_alg,
            load_alg,
        ]

        script_lines = generate_script_lines(raw_ws, ws_name)

        self.assertIn(mock.call(load_alg, []), get_alg_kwargs.call_args_list)
        self.assertEqual(get_alg_kwargs.call_count, 2)

        load_kwargs, output_ws = get_alg_kwargs(load_alg, ws_name)
        self.assertIn(
            "ws_{} = mc.{}({})\n".format(ws_name, load_alg.name(), load_kwargs),
            script_lines,
        )

    def test_that_get_algorithm_kwargs_works_as_expected_with_load(self):
        load_alg = mock.MagicMock()
        load_alg.name.return_value = "Load"

        load_alg_prop_filename = mock.MagicMock()
        load_alg_prop_filename.name.return_value = "Filename"
        load_alg_prop_filename.isDefault.return_value = False

        load_alg.getProperties.return_value = [load_alg_prop_filename]

        arguments, output_ws = get_algorithm_kwargs(load_alg, [])

        self.assertIn(
            "{}='{}'".format("Filename", load_alg_prop_filename.value()), arguments
        )

    def test_that_get_algorithm_kwargs_works_as_expected_with_make_projection(self):
        make_proj_alg = mock.MagicMock()
        make_proj_alg.name.return_value = "MakeProjection"

        make_proj_alg_prop_input_ws = mock.MagicMock()
        make_proj_alg_prop_input_ws.name.return_value = "InputWorkspace"
        make_proj_alg_prop_input_ws.value.return_value = "workspace_name"
        make_proj_alg_prop_input_ws.isDefault.return_value = False

        make_proj_alg_prop_output_ws = mock.MagicMock()
        make_proj_alg_prop_output_ws.name.return_value = "OutputWorkspace"
        make_proj_alg_prop_output_ws.value.return_value = "output_workspace_name"
        make_proj_alg_prop_output_ws.isDefault.return_value = False

        make_proj_alg.getProperties.return_value = [
            make_proj_alg_prop_output_ws,
            make_proj_alg_prop_input_ws,
        ]

        args, output_ws = get_algorithm_kwargs(make_proj_alg, ["workspace_name"])

        self.assertNotIn(
            "{}='{}'".format("OutputWorkspace", make_proj_alg_prop_output_ws.value()),
            args,
        )
        self.assertIn("{}={}".format("InputWorkspace", "workspace_name"), args)

    def test_that_get_algorithm_kwargs_works_as_expected_with_make_projection_using_string_property_value(
        self,
    ):
        make_proj_alg = mock.MagicMock()
        make_proj_alg.name.return_value = "MakeProjection"

        make_proj_alg_prop = mock.MagicMock()
        make_proj_alg_prop.name.return_value = "SomeProp"
        make_proj_alg_prop.isDefault.return_value = False
        make_proj_alg_prop.value.return_value = "string"

        make_proj_alg.getProperties.return_value = [make_proj_alg_prop]

        args, output_ws = get_algorithm_kwargs(make_proj_alg, "workspace_name")

        self.assertIn("{}='{}'".format("SomeProp", "string"), args)

    def test_that_get_algorithm_kwargs_works_as_expected_with_make_projection_using_non_string_property_value(
        self,
    ):
        make_proj_alg = mock.MagicMock()
        make_proj_alg.name.return_value = "MakeProjection"

        make_proj_alg_prop = mock.MagicMock()
        make_proj_alg_prop.name.return_value = "SomeProp"
        make_proj_alg_prop.isDefault.return_value = False
        make_proj_alg_prop.value.return_value = 12

        make_proj_alg.getProperties.return_value = [make_proj_alg_prop]

        args, output_ws = get_algorithm_kwargs(make_proj_alg, "workspace_name")

        self.assertIn("{}={}".format("SomeProp", 12), args)

    def test_that_get_algorithm_kwargs_works_as_expected_with_a_general_algorithm_using_non_string_property_value(
        self,
    ):
        some_alg = mock.MagicMock()
        some_alg.name.return_value = "SomeAlgorithm"

        some_alg_prop = mock.MagicMock()
        some_alg_prop.name.return_value = "SomeProp"
        some_alg_prop.isDefault.return_value = False
        some_alg_prop.value.return_value = 12

        some_alg.getProperties.return_value = [some_alg_prop]

        args, output_ws = get_algorithm_kwargs(some_alg, "workspace_name")

        self.assertIn("{}={}".format("SomeProp", 12), args)

    def test_that_get_algorithm_kwargs_works_as_expected_with_a_general_algorithm_using_string_property_value(
        self,
    ):
        some_alg = mock.MagicMock()
        some_alg.name.return_value = "SomeAlgorithm"

        some_alg_prop = mock.MagicMock()
        some_alg_prop.name.return_value = "SomeProp"
        some_alg_prop.isDefault.return_value = False
        some_alg_prop.value.return_value = "string"

        some_alg.getProperties.return_value = [some_alg_prop]

        args, output_ws = get_algorithm_kwargs(some_alg, "workspace_name")

        self.assertIn("{}='{}'".format("SomeProp", "string"), args)

    @mock.patch("qtpy.QtGui.QGuiApplication.clipboard")
    @mock.patch("builtins.open")
    def test_generate_script_produces_the_same_script_for_file_and_clipboard(
        self, mock_open, mock_clipboard
    ):
        workspace = self.create_workspace("test")
        add_workspace(workspace, workspace.name)

        plot_handler = mock.MagicMock()
        plot_window = plot_handler.plot_window

        fake_file = FakeFile()
        mock_open.return_value = fake_file

        generate_script(
            workspace.name,
            filename="filename",
            plot_handler=plot_handler,
            window=plot_window,
        )
        mock_open.assert_called_once()

        fake_clipboard = FakeClipboard()
        mock_clipboard.return_value = fake_clipboard

        generate_script(
            workspace.name,
            plot_handler=plot_handler,
            window=plot_window,
            clipboard=True,
        )
        mock_clipboard.assert_called_once()

        self.assertEqual(fake_clipboard.text, fake_file.text)

    def test_that_get_algorithm_kwargs_produces_raw_string_for_loaded_filename(self):
        some_alg = mock.MagicMock()
        some_alg.name.return_value = "Load"

        some_alg_prop = mock.MagicMock()
        some_alg_prop.name.return_value = "Filename"
        some_alg_prop.isDefault.return_value = False
        some_alg_prop.value.return_value = "test_filename"

        some_alg.getProperties.return_value = [some_alg_prop]

        args, _ = get_algorithm_kwargs(some_alg, "workspace_name")

        self.assertEqual("Filename=r'test_filename'", args)
