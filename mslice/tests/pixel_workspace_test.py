from __future__ import (absolute_import, division, print_function)
import numpy as np
import unittest

from mantid.simpleapi import CreateSimulationWorkspace, ConvertToMD, AddSampleLog

from mslice.workspace.pixel_workspace import PixelWorkspace


class PixelWorkspaceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):  # create (non-zero) test data
        sim_workspace = CreateSimulationWorkspace(Instrument='MAR', BinParams=[-10, 1, 10],
                                                  UnitX='DeltaE', OutputWorkspace='simws')
        AddSampleLog(sim_workspace, LogName='Ei', LogText='3.', LogType='Number')
        cls.workspace = ConvertToMD(InputWorkspace=sim_workspace, OutputWorkspace="convert_ws", QDimensions='|Q|',
                                    dEAnalysisMode='Direct', MinValues='-10,0,0', MaxValues='10,6,500',
                                    SplitInto='50,50')
        cls.workspace = PixelWorkspace(cls.workspace)

    def test_invalid_workspace(self):
        self.assertRaises(TypeError, lambda: PixelWorkspace(4))

    def test_get_coordinates(self):
        coords = self.workspace.get_coordinates()
        self.assertEqual(set(coords), {'|Q|', 'DeltaE'})
        self.assertEqual(coords['|Q|'][2], 0.20996594809147776)

    def test_get_signal(self):
        signal = self.workspace.get_signal()
        self.assertEqual(0, signal[0][0])
        self.assertEqual(12, signal[1][47])
        self.assertEqual(32, signal[3][52])

    def test_get_error(self):
        expected = np.zeros((100, 100))
        self.assertTrue((self.workspace.get_error() == expected).all())

    def test_get_variance(self):
        expected = np.zeros((100, 100))
        self.assertTrue((self.workspace.get_variance() == expected).all())

    def test_add_workspace(self):
        two_workspace = self.workspace + self.workspace
        signal = two_workspace.get_signal()
        self.assertEqual(0, signal[0][0])
        self.assertEqual(24, signal[1][47])
        self.assertEqual(64, signal[3][52])

    def test_mul_workspace_number(self):
        three_workspace = self.workspace * 3
        signal = three_workspace.get_signal()
        self.assertEqual(0, signal[0][0])
        self.assertEqual(36, signal[1][47])
        self.assertEqual(96, signal[3][52])

    def test_pow_workspace(self):
        squared_workspace = self.workspace ** 2
        signal = squared_workspace.get_signal()
        self.assertEqual(0, signal[0][0])
        self.assertEqual(144, signal[1][47])
        self.assertEqual(1024, signal[3][52])

    def test_neg_workspace(self):
        neg_workspace = -self.workspace
        signal = neg_workspace.get_signal()
        self.assertEqual(0, signal[0][0])
        self.assertEqual(-12, signal[1][47])
        self.assertEqual(-32, signal[3][52])

    def test_add_list(self):
        list_to_add = np.linspace(0, 99, 100)
        result = self.workspace + list_to_add
        result = result.get_signal()
        self.assertAlmostEqual(0, result[0][0], 8)
        self.assertAlmostEqual(13, result[1][47], 8)
        self.assertAlmostEqual(35, result[3][52], 8)
