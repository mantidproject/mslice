import unittest
from mock import MagicMock, patch
from mslice.models.axis import Axis, STEP_TOLERANCE


class AxisTest(unittest.TestCase):

    def test_success_int(self):
        axis = Axis('x', 0, 5, 1)
        self.assertEqual(axis.start, 0.0)
        self.assertTrue(isinstance(axis.start, float))
        self.assertEqual(axis.end, 5.0)
        self.assertTrue(isinstance(axis.end, float))
        self.assertEqual(axis.step, 1.0)
        self.assertTrue(isinstance(axis.end, float))

    def test_success_string(self):
        axis = Axis('x', '0', '5', '1')
        self.assertEqual(axis.start, 0.0)
        self.assertTrue(isinstance(axis.start, float))
        self.assertEqual(axis.end, 5.0)
        self.assertTrue(isinstance(axis.end, float))
        self.assertEqual(axis.step, 1.0)
        self.assertTrue(isinstance(axis.end, float))

    def test_invalid_string_start(self):
        with self.assertRaises(ValueError):
            Axis('x', 'aa', '1', '.1')

    def test_invalid_string_end(self):
        with self.assertRaises(ValueError):
            Axis('x', '0', 'aa', '.1')

    def test_invalid_string_step(self):
        with self.assertRaises(ValueError):
            Axis('x', '0', '1', 'aa')

    def test_invalid_start_greater_than_end(self):
        with self.assertRaises(ValueError):
            Axis('x', '1', '0', '.1')

    def test_energy_units(self):
        with self.assertRaises(ValueError):
            Axis('DeltaE', '0', '5', '1', 'not_an_energy')
        axis = Axis('DeltaE', '0', '5', '1', 'cm-1')
        self.assertNotEqual(axis.scale, 1)
        self.assertAlmostEqual(axis.start_meV, axis.start * axis.scale, 3)
        self.assertAlmostEqual(axis.end_meV, axis.end * axis.scale, 3)
        self.assertAlmostEqual(axis.step_meV, axis.step * axis.scale, 3)

    @patch('mslice.models.axis.get_axis_step')
    def test_validate_step_against_workspace_returns_none_if_the_cut_width_is_larger_than_the_data_x_width(self, mock_get_axis_step):
        mock_ws = MagicMock()
        mock_get_axis_step.return_value = 0.02
        axis = Axis('DeltaE', '0', '5', '0.1', 'cm-1')
        self.assertEqual(None, axis.validate_step_against_workspace(mock_ws))

    @patch('mslice.models.axis.get_axis_step')
    def test_validate_step_against_workspace_returns_error_if_the_cut_width_is_smaller_than_the_data_x_width(self, mock_get_axis_step):
        mock_ws = MagicMock()
        mock_get_axis_step.return_value = 0.2
        axis = Axis('DeltaE', '0', '5', '0.1', 'cm-1')
        self.assertEqual('The DeltaE step provided (0.1000) is smaller than the data step in the '
                         'workspace (0.2000). Please provide a larger DeltaE step.', axis.validate_step_against_workspace(mock_ws))

    @patch('mslice.models.axis.get_axis_step')
    def test_validate_step_against_workspace_returns_none_if_the_cut_width_is_smaller_than_the_data_x_width_but_in_tol(self,
                                                                                                                       mock_get_axis_step):
        mock_ws = MagicMock()
        mock_get_axis_step.return_value = 0.1 + STEP_TOLERANCE
        axis = Axis('DeltaE', '0', '5', '0.1', 'cm-1')
        self.assertEqual(None, axis.validate_step_against_workspace(mock_ws))
