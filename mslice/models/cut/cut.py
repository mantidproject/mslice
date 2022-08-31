from mslice.models.intensity_correction_algs import (compute_chi, compute_d2sigma,
                                                     compute_symmetrised)
from mslice.models.labels import is_momentum, is_twotheta
from mslice.util.intensity_correction import IntensityType

import numpy as np


class Cut(object):
    """Groups parameters needed to cut and validates them, caches intensities"""

    def __init__(self, cut_axis, integration_axis, intensity_start, intensity_end, norm_to_one=False, width=None,
                 algorithm='Rebin', sample_temp=None, e_fixed=None):
        self._cut_ws = None
        self.cut_axis = cut_axis
        self.integration_axis = integration_axis
        self.intensity_start = intensity_start
        self.intensity_end = intensity_end
        self.norm_to_one = norm_to_one
        self.icut = None
        # These are here to save the integration range as the integration axis is
        # changed when a cut with a width is performed
        self.start = integration_axis.start
        self.end = integration_axis.end
        self.width = width
        self.algorithm = algorithm
        self._sample_temp = sample_temp
        self._e_fixed = e_fixed
        self._corrected_intensity_range_cache = {}
        self.parent_ws_name = None

        # intensities
        self._chi = None
        self._chi_magnetic = None
        self._d2sigma = None
        self._symmetrised = None

    @property
    def cut_ws(self):
        return self._cut_ws

    @cut_ws.setter
    def cut_ws(self, cut_ws):
        self._cut_ws = cut_ws
        if self._cut_ws:
            self._update_cut_axis()

    def reset_integration_axis(self, start, end):
        self.integration_axis.start = start
        self.integration_axis.end = end

    @property
    def workspace_name(self):
        return self._cut_ws.name

    @property
    def workspace_raw_name(self):
        return self._cut_ws.raw_ws.name()

    @property
    def cut_axis(self):
        return self._cut_axis

    @cut_axis.setter
    def cut_axis(self, axis):
        self._cut_axis = axis

    @property
    def integration_axis(self):
        return self._integration_axis

    @integration_axis.setter
    def integration_axis(self, axis):
        self._integration_axis = axis

    @property
    def intensity_start(self):
        return self._intensity_start

    @intensity_start.setter
    def intensity_start(self, int_start):
        if int_start is None:
            self._intensity_start = None
        else:
            try:
                self._intensity_start = None if int_start == '' else float(int_start)
            except ValueError:
                raise ValueError('Invalid intensity parameters')

    @property
    def intensity_end(self):
        return self._intensity_end

    @intensity_end.setter
    def intensity_end(self, int_end):
        if int_end is None:
            self._intensity_end = None
        else:
            try:
                self._intensity_end = None if int_end == '' else float(int_end)
            except ValueError:
                raise ValueError('Invalid intensity parameters')

    @property
    def norm_to_one(self):
        return self._norm_to_one

    @norm_to_one.setter
    def norm_to_one(self, value):
        self._norm_to_one = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width_str):
        if width_str is not None and width_str.strip():
            if width_str.startswith('e') or width_str.endswith('e') or width_str.startswith('-'):
                self._width = None
            else:
                try:
                    self._width = float(width_str)
                except ValueError:
                    raise ValueError("Invalid width")
        elif width_str == '':
            self._width = self.end - self.start
        else:
            self._width = None

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algo):
        self._algorithm = algo

    @property
    def raw_sample_temp(self):
        return self._sample_temp

    @property
    def sample_temp(self):
        if self._sample_temp is None:
            raise SampleTempValueError('sample temperature not found', self.workspace_name)
        return self._sample_temp

    @sample_temp.setter
    def sample_temp(self, value):
        self._sample_temp = value

    @property
    def chi(self):
        if self._chi is None:
            self._chi = compute_chi(self._cut_ws, self.sample_temp, self.e_axis)
            self._chi.intensity_corrected = True
            self._corrected_intensity_range_cache[IntensityType.CHI] = self._get_intensity_range_from_ws(self._chi)
        return self._chi

    @property
    def chi_magnetic(self):
        if self._chi_magnetic is None:
            self._chi_magnetic = compute_chi(self._cut_ws, self.sample_temp, self.e_axis, True)
            self._chi_magnetic.intensity_corrected = True
            self._corrected_intensity_range_cache[IntensityType.CHI_MAGNETIC] = \
                self._get_intensity_range_from_ws(self._chi_magnetic)
        return self._chi_magnetic

    @property
    def d2sigma(self):
        if self._d2sigma is None:
            self._d2sigma = compute_d2sigma(self._cut_ws, self.e_axis, self._e_fixed)
            self._d2sigma.intensity_corrected = True
            self._corrected_intensity_range_cache[IntensityType.D2_SIGMA] = self._get_intensity_range_from_ws(self._d2sigma)
        return self._d2sigma

    @property
    def symmetrised(self):
        if self._symmetrised is None:
            self._symmetrised = compute_symmetrised(self._cut_ws, self.sample_temp, self.e_axis, self.rotated)
            self._symmetrised.intensity_corrected = True
            self._corrected_intensity_range_cache[IntensityType.SYMMETRISED] = \
                self._get_intensity_range_from_ws(self._symmetrised)
        return self._symmetrised

    def get_intensity_corrected_ws(self, intensity_correction_type):
        if intensity_correction_type == IntensityType.SCATTERING_FUNCTION:
            return self._cut_ws
        elif intensity_correction_type == IntensityType.CHI:
            return self.chi
        elif intensity_correction_type == IntensityType.CHI_MAGNETIC:
            return self.chi_magnetic
        elif intensity_correction_type == IntensityType.D2_SIGMA:
            return self.d2sigma
        elif intensity_correction_type == IntensityType.SYMMETRISED:
            return self.symmetrised

    @property
    def e_axis(self):
        if self.rotated:
            return self.cut_axis
        else:
            return self.integration_axis

    @property
    def q_axis(self):
        if self.rotated:
            return self.integration_axis
        else:
            return self.cut_axis

    def _update_cut_axis(self):
        x_dim = self._cut_ws.raw_ws.getXDimension()
        if self.cut_axis.units == x_dim.getUnits() or self.cut_axis.units in x_dim.getDimensionId():
            ws_cut_axis = x_dim
        else:
            ws_cut_axis = self._cut_ws.raw_ws.getYDimension()
        self.cut_axis.step = ws_cut_axis.getBinWidth()
        self.cut_axis.start = ws_cut_axis.getMinimum()
        self.cut_axis.end = ws_cut_axis.getMaximum()

    @property
    def intensity_corrected(self):
        return self.cut_ws.intensity_corrected

    def get_corrected_intensity_range(self, intensity_correction):
        return self._corrected_intensity_range_cache[intensity_correction]

    @staticmethod
    def _get_intensity_range_from_ws(workspace):
        min_intensity = np.nanmin(workspace.get_signal())
        max_intensity = np.nanmax(workspace.get_signal())
        return min_intensity, max_intensity

    def copy_for_cache(self):
        cut_axis, int_axis = tuple(self.cut_ws.axes)
        cut = Cut(cut_axis, int_axis, self.intensity_start, self.intensity_end, self.norm_to_one,
                  '', self.algorithm, self._sample_temp, self._e_fixed)
        cut._cut_ws = self.cut_ws
        cut.parent_ws_name = self.parent_ws_name
        return cut

    @property
    def rotated(self):
        return not is_twotheta(self.cut_axis.units) and not is_momentum(self.cut_axis.units)


class SampleTempValueError(ValueError):
    def __init__(self, message, ws_name):
        self.message = message
        self.ws_name = ws_name
