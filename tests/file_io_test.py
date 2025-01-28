from mock import patch, Mock, MagicMock
import numpy as np
from os.path import join
from tempfile import gettempdir
import unittest
from mantid.simpleapi import CreateMDHistoWorkspace
from mantid.kernel import ConfigService

from mslice.models.axis import Axis
from mslice.models.cut.cut_functions import output_workspace_name
from mslice.models.workspacemanager.file_io import (
    _save_cut_to_ascii,
    _save_slice_to_ascii,
    get_save_directory,
    _to_absolute_path,
)
from mslice.workspace.histogram_workspace import HistogramWorkspace


class FileIOTest(unittest.TestCase):
    def setUp(self):
        self.mock_dialog = MagicMock()
        self.tempdir = gettempdir()
        self.path = join(self.tempdir, "some_file")
        self.mock_dialog.selectedFiles.return_value = [self.path]

    @patch("mslice.models.workspacemanager.file_io.QFileDialog")
    def test_save_directory_default(self, file_dialog_mock):
        file_dialog_mock.return_value = self.mock_dialog
        self.mock_dialog.selectedFilter.return_value = "Nexus (*.nxs)"

        directory, file_name, extension = get_save_directory(default_ext=".nxs")
        self.mock_dialog.setNameFilter.assert_called_once_with(
            "Nexus (*.nxs);; NXSPE (*.nxspe);; Ascii (*.txt);; Matlab (*.mat)"
        )
        self.mock_dialog.selectNameFilter.assert_called_once_with("Nexus (*.nxs)")
        self.mock_dialog.exec_.assert_called_once()
        self.assertEqual(directory, self.tempdir)
        self.assertEqual(file_name, "some_file.nxs")
        self.assertEqual(extension, ".nxs")

    @patch("mslice.models.workspacemanager.file_io.QFileDialog")
    def test_save_directory_image(self, file_dialog_mock):
        file_dialog_mock.return_value = self.mock_dialog
        self.mock_dialog.selectedFilter.return_value = "Image (*.png)"

        directory, file_name, extension = get_save_directory(save_as_image=True)
        self.mock_dialog.setNameFilter.assert_called_once_with(
            "Image (*.png);; PDF (*.pdf);; Nexus (*.nxs);; NXSPE (*.nxspe);; Ascii (*.txt);; Matlab (*.mat)"
        )
        self.mock_dialog.exec_.assert_called_once()
        self.assertEqual(directory, self.tempdir)
        self.assertEqual(file_name, "some_file.png")
        self.assertEqual(extension, ".png")

    @patch("mslice.models.workspacemanager.file_io.QFileDialog")
    def test_save_directory_multiple(self, file_dialog_mock):
        file_dialog_mock.getExistingDirectory = Mock(return_value=self.path)

        directory, file_name, extension = get_save_directory(
            multiple_files=True, default_ext=".nxs"
        )
        self.assertEqual(directory, self.path)
        self.assertEqual(file_name, None)
        self.assertEqual(extension, ".nxs")
        self.mock_dialog.exec_.assert_not_called()

    @patch("mslice.models.workspacemanager.file_io.QFileDialog")
    def test_save_directory_double_extension(self, file_dialog_mock):
        file_dialog_mock.return_value = self.mock_dialog
        self.path = join(self.tempdir, "some_file.nxs")
        self.mock_dialog.selectedFiles.return_value = [self.path]

        directory, file_name, extension = get_save_directory()
        self.mock_dialog.exec_.assert_called_once()
        self.assertEqual(directory, self.tempdir)
        self.assertEqual(file_name, "some_file.nxs")
        self.assertEqual(extension, ".nxs")

    @patch("mslice.models.workspacemanager.file_io.QFileDialog")
    def test_save_directory_cancelled(self, file_dialog_mock):
        file_dialog_mock.return_value = self.mock_dialog
        self.mock_dialog.exec_.return_value = False
        self.mock_dialog.selectedFiles.assert_not_called()
        self.mock_dialog.selectedFilter.assert_not_called()

    @patch("mslice.models.workspacemanager.file_io._output_data_to_ascii")
    def test_save_cut_to_ascii(self, output_method):
        ws_name = output_workspace_name("workspace", -1.5, 2)
        raw_ws = CreateMDHistoWorkspace(
            SignalInput=[1, 2],
            ErrorInput=[4, 5],
            Dimensionality=1,
            Extents=[-1, 5],
            NumberOfBins=2,
            Names="Dim1",
            Units="units",
            OutputWorkspace=ws_name,
        )
        ws = HistogramWorkspace(raw_ws, ws_name)
        ws.axes = [Axis("DeltaE", 0, 1, 0.1), Axis("Q", 0, 2, 0.2)]
        _save_cut_to_ascii(ws, "some_path")
        output_method.assert_called_once()
        self.assertEqual(output_method.call_args[0][0], "some_path")
        np.testing.assert_array_equal(
            output_method.call_args[0][1], [[-1, 1, 4], [5, 2, 5]]
        )

    @patch("mslice.models.workspacemanager.file_io._output_data_to_ascii")
    def test_save_slice_to_ascii(self, output_method):
        raw_ws = CreateMDHistoWorkspace(
            SignalInput=[1, 2, 3, 4],
            ErrorInput=[3, 4, 5, 6],
            Dimensionality=2,
            Extents=[-10, 10, -10, 10],
            NumberOfBins="2,2",
            Names="Dim1,Dim2",
            Units="units1,units2",
            OutputWorkspace="workspace",
        )
        ws = HistogramWorkspace(raw_ws, "workspace")
        ws.axes = [Axis("DeltaE", 0, 1, 0.1), Axis("Q", 0, 2, 0.2)]
        _save_slice_to_ascii(ws, "path1")
        output_method.assert_called_once()
        self.assertEqual(output_method.call_args[0][0], "path1")
        np.testing.assert_array_equal(
            output_method.call_args[0][1],
            [[-10, -10, 1, 3], [-10, 10, 3, 5], [10, -10, 2, 4], [10, 10, 4, 6]],
        )

    def test_to_absolute_path_returns_the_same_path_if_it_is_already_absolute(self):
        abs_path = "/c/user/documents/filename.txt"
        self.assertEqual(abs_path, _to_absolute_path(abs_path))

    def test_to_absolute_path_returns_a_path_in_the_default_save_dir_if_the_path_provided_is_relative(
        self,
    ):
        default_save_directory = "/d/user/documents/default_save_dir"
        ConfigService.setString("defaultsave.directory", default_save_directory)

        rel_path = "datadir/filename.txt"
        self.assertEqual(
            join(default_save_directory, rel_path), _to_absolute_path(rel_path)
        )
