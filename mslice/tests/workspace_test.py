import unittest
import numpy as np
from mantid.simpleapi import CreateWorkspace, CreateMDWorkspace, CreateMDHistoWorkspace

from mslice.workspace.workspace import Workspace
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.histogram_workspace import HistogramWorkspace


class BaseWorkspaceTest(unittest.TestCase):

    def check_signal(self):
        expected_values = np.arange(0, 100)
        result = np.array(self.workspace.get_signal().flatten())
        result.sort()
        self.assertTrue((result == expected_values).all())

    def check_error(self):
        expected = np.zeros(100) + 2
        self.assertTrue((expected == self.workspace.get_error().flatten()).all())

    def check_variance(self):
        expected = np.zeros(100) + 4
        self.assertTrue((expected == self.workspace.get_variance().flatten()).all())


class WorkspaceTest(BaseWorkspaceTest):

    def setUp(self):
        x = np.linspace(0, 99, 100)
        y = x * 1
        e = y * 0 + 2
        self.workspace = Workspace(CreateWorkspace(x, y, e, OutputWorkspace="testBaseWorkspace"))

    def test_get_coordinates(self):
        expected_values = np.linspace(0, 99, 100)
        self.assertTrue((expected_values == self.workspace.get_coordinates()['']).all())

    def test_get_signal(self):
        expected_values = range(0, 100)
        result = np.array(self.workspace.get_signal().flatten())
        self.assertTrue((result == expected_values).all())

    def test_get_error(self):
        self.check_error()

    def test_get_variance(self):
        self.check_variance()


class PixelWorkspaceTest(BaseWorkspaceTest):

    def setUp(self):
        self.workspace = PixelWorkspace(CreateMDWorkspace(dimensions=2, Extents='0,100,0,100', names='Dim1,Dim2',
                                                          units='U,U', OutputWorkspace='testPixelWorkspace'))

    def test_get_coordinates(self):  # checks keys
        self.assertTrue(set(self.workspace.get_coordinates()) == {'Dim1', 'Dim2'})

    def test_get_signal(self):
        expected = np.zeros((100, 100))
        self.assertTrue((self.workspace.get_signal() == expected).all())

    def test_get_error(self):
        expected = np.zeros((100, 100))
        self.assertTrue((self.workspace.get_error() == expected).all())

    def test_get_variance(self):
        expected = np.zeros((100, 100))
        self.assertTrue((self.workspace.get_variance() == expected).all())


class HistogramWorkspaceTest(BaseWorkspaceTest):

    def setUp(self):
        signal = range(0, 100)
        error = np.zeros(100) + 2
        self.workspace = HistogramWorkspace(CreateMDHistoWorkspace(Dimensionality=2, Extents='0,100,0,100',
                                                                   SignalInput=signal, ErrorInput=error,
                                                                   NumberOfBins='10,10', Names='Dim1,Dim2',
                                                                   Units='U,U', OutputWorkspace='testHistoWorkspace'))

    def test_get_coordinates(self):
        expected = np.linspace(0, 100, 10)
        self.assertTrue((self.workspace.get_coordinates()['Dim1'] == expected).all())

    def test_get_signal(self):
        self.check_signal()

    def test_get_error(self):
        self.check_error()

    def test_get_variance(self):
        self.check_variance()
