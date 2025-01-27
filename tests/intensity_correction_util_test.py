from mock import patch, MagicMock
import unittest

from mslice.util.intensity_correction import IntensityCache, IntensityType
from mslice.models.axis import Axis
from mslice.models.cut.cut import Cut
from mslice.models.slice.slice import Slice
from mslice.plotting.plot_window.plot_window import PlotWindow
from mslice.plotting.pyplot import CATEGORY_SLICE, CATEGORY_CUT
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter


class IntensityCorrectionUtilTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enums = [
            IntensityType.SCATTERING_FUNCTION,
            IntensityType.CHI,
            IntensityType.CHI_MAGNETIC,
            IntensityType.SYMMETRISED,
            IntensityType.D2SIGMA,
            IntensityType.GDOS,
        ]

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

    def test_cache_action(self):
        IntensityCache._IntensityCache__action_dict = {}
        intensity_correction_type = IntensityType.SCATTERING_FUNCTION
        action = MagicMock
        IntensityCache.cache_action(intensity_correction_type, action)
        returned_action = IntensityCache.get_action(intensity_correction_type)
        self.assertEqual(action, returned_action)

    def test_cache_method(self):
        IntensityCache._IntensityCache__method_dict_slice = {}
        category = CATEGORY_SLICE
        intensity_correction_type = IntensityType.CHI_MAGNETIC
        method = MagicMock
        IntensityCache.cache_method(category, intensity_correction_type, method)
        returned_method = IntensityCache.get_method(category, intensity_correction_type)
        self.assertEqual(method, returned_method)

    def test_methods_cached_by_cut_plotter_presenter(self):
        IntensityCache._IntensityCache__method_dict_slice = {}
        presenter = CutPlotterPresenter()
        enums = [
            IntensityType.SCATTERING_FUNCTION,
            IntensityType.CHI,
            IntensityType.CHI_MAGNETIC,
            IntensityType.SYMMETRISED,
            IntensityType.D2SIGMA,
            IntensityType.GDOS,
        ]
        methods = [IntensityCache.get_method(CATEGORY_CUT, e) for e in enums]
        for method in methods:
            self.assertTrue(hasattr(presenter, method))

    @patch("mslice.plotting.plot_window.plot_window.add_action")
    @patch("mslice.plotting.plot_window.plot_window.PlotWindow.setup_ui")
    @patch("mslice.plotting.plot_window.plot_window.PlotWindow._PlotWindow__inherit")
    def test_actions_cached_by_cut_plotter_presenter(
        self, mock_inherit, mock_setup_ui, mock_add_action
    ):
        IntensityCache._IntensityCache__action_dict = {}
        plot_window = PlotWindow(MagicMock())
        plot_window.add_intensity_actions(MagicMock())
        enums = [
            IntensityType.SCATTERING_FUNCTION,
            IntensityType.CHI,
            IntensityType.CHI_MAGNETIC,
            IntensityType.SYMMETRISED,
            IntensityType.D2SIGMA,
            IntensityType.GDOS,
        ]
        actions = [IntensityCache.get_action(e) for e in enums]
        for action in actions:
            self.assertTrue(hasattr(plot_window, action))
        mock_setup_ui.assert_called_once()

    @staticmethod
    def _create_slice(
        workspace=None,
        colourmap=None,
        norm=None,
        sample_temp=None,
        q_axis=None,
        e_axis=None,
        rotated=None,
    ):
        return Slice(workspace, colourmap, norm, sample_temp, q_axis, e_axis, rotated)

    @staticmethod
    def _create_cut(
        q_axis=Axis("|Q|", 0.1, 3.1, 0.1),
        e_axis=Axis("DeltaE", -10, 15, 1),
        intensity_start=0,
        intensity_end=100,
        norm_to_one=False,
        width=None,
        algorithm="Rebin",
        sample_temp=None,
        e_fixed=None,
    ):
        return Cut(
            e_axis,
            q_axis,
            intensity_start,
            intensity_end,
            norm_to_one,
            width,
            algorithm,
            sample_temp,
            e_fixed,
        )
