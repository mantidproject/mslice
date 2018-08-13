import unittest
from mslice.models.axis import Axis

class AxisTest(unittest.TestCase):

    def test_success_int(self):
        axis = Axis('x', 0, 5, 1)
        self.assertEqual(axis.start, 0.0)
        assert(isinstance(axis.start, float))
        self.assertEqual(axis.end, 5.0)
        assert(isinstance(axis.end, float))
        self.assertEqual(axis.step, 1.0)
        assert(isinstance(axis.end, float))

    def test_success_string(self):
        axis = Axis('x', '0', '5', '1')
        self.assertEqual(axis.start, 0.0)
        assert(isinstance(axis.start, float))
        self.assertEqual(axis.end, 5.0)
        assert(isinstance(axis.end, float))
        self.assertEqual(axis.step, 1.0)
        assert(isinstance(axis.end, float))

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





