import unittest
from mslice.models.units import EnergyUnits
from six import string_types

class EnergyUnitsTest(unittest.TestCase):

    def test_success(self):
        en_unit = EnergyUnits('meV')
        # the "meV" unit *must* be defined by this class as it is used in
        # all algorithms used by MSlice. Other units may be defined as desired
        self.assertEqual(en_unit.factor_from_meV(), 1.)
        self.assertEqual(en_unit.factor_to_meV(), 1.)
        val = list(en_unit.convert_from('meV',1.))[0]
        assert(isinstance(val, string_types))
        self.assertAlmostEqual(float(val),  1., 3)
        val = list(en_unit.convert_to('meV',1.))[0]
        assert(isinstance(val, string_types))
        self.assertAlmostEqual(float(val),  1., 3)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            EnergyUnits('not_an_energy_unit')
