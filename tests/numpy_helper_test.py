import unittest
import numpy as np

from mslice.util.numpy_helper import is_real_number


class NumpyHelperTest(unittest.TestCase):

    def test_is_real_number_returns_false_when_provided_none(self):
        self.assertFalse(is_real_number(None))

    def test_is_real_number_returns_false_when_provided_a_string_that_cannot_be_converted_to_float(self):
        self.assertFalse(is_real_number("34eta"))

    def test_is_real_number_returns_false_when_provided_an_inf_or_nan(self):
        self.assertFalse(is_real_number(np.inf))
        self.assertFalse(is_real_number(np.nan))

    def test_is_real_number_returns_true_for_a_valid_real_number(self):
        self.assertTrue(is_real_number("1.234"))
        self.assertTrue(is_real_number(5.678))

    def test_is_real_number_returns_true_for_a_valid_real_number_in_scientific_notation(self):
        self.assertTrue(is_real_number("1e5"))
        self.assertTrue(is_real_number(2e6))
