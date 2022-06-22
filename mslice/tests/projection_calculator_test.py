from unittest import TestCase

from mantid.api import AnalysisDataService

from mslice.models.projection.powder.mantid_projection_calculator import MantidProjectionCalculator
from mslice.tests.testhelpers.workspace_creator import create_simulation_workspace
from mslice.workspace.pixel_workspace import PixelWorkspace


class ProjectionCalculatorTest(TestCase):

    def setUp(self):
        self.non_psd_workspace = create_simulation_workspace("Direct", "non_psd_ws", psd=False)
        self.psd_workspace = create_simulation_workspace("Direct", "psd_ws", psd=True)
        self.psd_workspace.limits['MomentumTransfer'] = [0.1, 3.1, 0.1]
        self.projection_calculator = MantidProjectionCalculator()

    def tearDown(self) -> None:
        AnalysisDataService.clear()

    def test_available_axes(self):
        self.assertEqual(self.projection_calculator.available_axes(), ['|Q|', '2Theta', 'DeltaE'])

    def test_validate_workspace(self):
        self.assertEqual(self.projection_calculator.validate_workspace(self.psd_workspace), None)

    def test_projection_calculation(self):
        self.assertRaises(RuntimeError, self.projection_calculator.calculate_projection, self.non_psd_workspace, '|Q|',
                          'DeltaE')
        self.assertRaises(NotImplementedError, self.projection_calculator.calculate_projection, self.psd_workspace, '',
                          '')
        result = self.projection_calculator.calculate_projection(self.psd_workspace, '|Q|', 'DeltaE')
        self.assertEqual(type(result), PixelWorkspace)
