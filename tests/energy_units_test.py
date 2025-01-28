import unittest
from mslice.models.units import EnergyUnits


class EnergyUnitsTest(unittest.TestCase):
    def test_success(self):
        en_unit = EnergyUnits("meV")
        # the "meV" unit *must* be defined by this class as it is used in
        # all algorithms used by MSlice. Other units may be defined as desired
        self.assertEqual(en_unit.factor_from_meV(), 1.0)
        self.assertEqual(en_unit.factor_to_meV(), 1.0)
        val = list(en_unit.convert_from("meV", 1.0))[0]
        self.assertTrue(isinstance(val, str))
        self.assertAlmostEqual(float(val), 1.0, 3)
        val = list(en_unit.convert_to("meV", 1.0))[0]
        self.assertTrue(isinstance(val, str))
        self.assertAlmostEqual(float(val), 1.0, 3)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            EnergyUnits("not_an_energy_unit")
