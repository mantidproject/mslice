import unittest
from mslice.models.axis import Axis


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
