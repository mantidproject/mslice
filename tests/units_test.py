import unittest

from mslice.models.units import (
    EnergyUnits,
    _scale_string_or_float,
    get_sample_temperature_from_string,
)


class UnitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.en_unit = EnergyUnits("cm-1")

    def test_scale_string_or_float_error_check(self):
        self.assertEqual(_scale_string_or_float("NotAValue", 1.0), "NotAValue")

    def test_get_sample_temperature_from_string(self):
        self.assertEqual(get_sample_temperature_from_string("100K"), 100.0)
        self.assertEqual(get_sample_temperature_from_string("100"), 100.0)
        self.assertEqual(get_sample_temperature_from_string("NotATemperature"), None)

    def test_get_index_of_energy_unit(self):
        self.assertEqual(self.en_unit.index(), 1)

    def test_to_mev_conversion(self):
        result = list(x for x in self.en_unit.to_meV(1.0))
        self.assertAlmostEqual("0.12398", result[0], 4)

    def test_factor_from_with_unrecognised_unit(self):
        self.assertRaises(ValueError, self.en_unit.factor_from, "K")

    def test_factor_to_with_unrecognised_unit(self):
        self.assertRaises(ValueError, self.en_unit.factor_to, "K")

    def test_get_index(self):
        self.assertEqual(self.en_unit.get_index("meV"), 0)

    def test_get_index_with_unrecognised_unit(self):
        self.assertRaises(ValueError, self.en_unit.get_index, "K")
