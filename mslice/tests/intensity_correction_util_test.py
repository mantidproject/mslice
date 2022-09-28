from __future__ import (absolute_import, division, print_function)

from mock import patch, MagicMock
import unittest

from mslice.util.intensity_correction import IntensityCache, IntensityType
from mslice.models.axis import Axis
from mslice.models.cut.cut import Cut
from mslice.models.slice.slice import Slice


class IntensityCorrectionUtilTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enums = [IntensityType.SCATTERING_FUNCTION, IntensityType.CHI, IntensityType.CHI_MAGNETIC,
                     IntensityType.SYMMETRISED, IntensityType.D2SIGMA, IntensityType.GDOS]

    def test_enum_conversion(self):
        descs = [IntensityCache.get_desc_from_type(e) for e in self.enums]
        final_enums = [IntensityCache.get_intensity_type_from_desc(d) for d in descs]
        self.assertEqual(self.enums, final_enums)

    def test_enum_name_is_model_method(self):
        slice = self._create_slice()
        cut = self._create_cut()
        slice_attrs = slice.__dir__()
        cut_attrs = cut.__dir__()

        for e in self.enums:
            if e is not IntensityType.SCATTERING_FUNCTION:
                self.assertTrue(slice_attrs.count(e.name.lower()))
                self.assertTrue(cut_attrs.count(e.name.lower()))

    @staticmethod
    def _create_slice(workspace=None, colourmap=None, norm=None, sample_temp=None,
                      q_axis=None, e_axis=None, rotated=None):
        return Slice(workspace, colourmap, norm, sample_temp, q_axis, e_axis, rotated)

    @staticmethod
    def _create_cut(q_axis=Axis("|Q|", 0.1, 3.1, 0.1), e_axis=Axis("DeltaE", -10, 15, 1), intensity_start=0,
                    intensity_end=100, norm_to_one=False, width=None, algorithm='Rebin', sample_temp=None, e_fixed=None):
        return Cut(e_axis, q_axis, intensity_start, intensity_end, norm_to_one, width, algorithm, sample_temp, e_fixed)
