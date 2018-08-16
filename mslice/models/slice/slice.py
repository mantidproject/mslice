from mslice.models.slice.slice_functions import (compute_chi, compute_chi_magnetic, compute_d2sigma, compute_gdos,
                                                 compute_symmetrised)


class Slice():
    """class that caches intensities and parameters for a single workspace"""
    def __init__(self, slice, colourmap, norm, sample_temp, q_axis, e_axis, rotated=False):
        """
        :param slice: output workspace of Slice algorithm
        :param colourmap: default colourmap to use
        :param norm: normalisation - contains min and max intensity values
        :param sample_temp: cached sample temperature
        :param q_axis: non-energy Axis object (momentum or scattering angle)
        :param e_axis: energy Axis object
        :param rotated: true if energy is on the x axis
        """
        self.scattering_function = slice
        self.colourmap = colourmap
        self.norm = norm
        self._sample_temp = sample_temp
        self.momentum_axis = q_axis
        self.energy_axis = e_axis
        self.rotated = rotated
        self.overplot_lines = {}

        #intensities
        self._chi = None
        self._chi_magnetic = None
        self._d2sigma = None
        self._symmetrised = None
        self._gdos = None


    @property
    def sample_temp(self):
        if self._sample_temp is None:
            raise ValueError('sample temperature not found')
        return self._sample_temp

    @sample_temp.setter
    def sample_temp(self, value):
        self._sample_temp = value

    @property
    def chi(self):
        if self._chi is None:
            self._chi = compute_chi(self.scattering_function, self.sample_temp, self.energy_axis)
        return self._chi

    @property
    def chi_magnetic(self):
        if self._chi_magnetic is None:
            self._chi_magnetic = compute_chi_magnetic(self.chi)
        return self._chi_magnetic

    @property
    def d2sigma(self):
        if self._d2sigma is None:
            self._d2sigma = compute_d2sigma(self.scattering_function, self.energy_axis)
        return self._d2sigma

    @property
    def symmetrised(self):
        if self._symmetrised is None:
            self._symmetrised = compute_symmetrised(self.scattering_function, self.sample_temp, self.energy_axis, self.rotated)
        return self._symmetrised

    @property
    def gdos(self):
        if self._gdos is None:
            self._gdos = compute_gdos(self.scattering_function, self.sample_temp, self.momentum_axis, self.energy_axis)
        return self._gdos
