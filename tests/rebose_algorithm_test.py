import unittest

from mantid.api import AnalysisDataService
from tests.testhelpers.workspace_creator import create_workspace
from mslice.util.mantid.mantid_algorithms import Rebose


class RebinAlgorithmTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_workspace = create_workspace('test_workspace')

    @classmethod
    def tearDownClass(cls) -> None:
        AnalysisDataService.clear()

    def test_rebose_algorithm(self):
        results = Rebose(self.test_workspace)
        signal = results.get_signal()
        self.assertAlmostEqual(0.02571387, signal[0][0], 7)
        self.assertAlmostEqual(0.69316964, signal[1][17], 7)
        self.assertAlmostEqual(0.32440028, signal[3][26], 7)
        error = results.get_error()
        self.assertAlmostEqual(0.02249475, error [0][0], 7)
        self.assertAlmostEqual(0.22030317, error [1][17], 7)
        self.assertAlmostEqual(0.18108407, error [3][26], 7)
