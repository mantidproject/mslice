"""
Module for handling units.
At present only the energy units "meV" ("DeltaE") and "cm-1" ("DeltaE_inWavenumber")
are defined, but they are separated into a module for possible future expansion.
"""

from __future__ import (absolute_import, division, print_function)
from mslice.util import MPL_COMPAT

import numpy as np


def _scale_string_or_float(value, scale):
    try:
        return '{:.5f}'.format(float(value) * scale)
    except (ValueError, TypeError):
        return value


def get_sample_temperature_from_string(string):
    if string is not None and string.strip():
        if string.endswith('K'):
            string = string[:-1]
        try:
            sample_temp = float(string)
            return sample_temp
        except ValueError:
            return None
    return None


class EnergyUnits(object):

    _available_units = ['meV', 'cm-1']
    _label_latex = {'meV':'meV', 'cm-1':'cm$^{-1}$'}
    _name_to_index = {'meV':0, 'cm-1':1, 'DeltaE':0, 'DeltaE_inWavenumber':1}
    # The following is a conversion matrix between the different units e_to = m[from][to] * e_from
    # E.g. meV = m[1][0] * cm; and cm = m[0][1] * meV
    _conversion_factors = [[1.,    8.065544],
                           [1./8.065544, 1.]]

    def __init__(self, unit_name):
        if unit_name not in self._name_to_index.keys():
            raise ValueError("Unrecognised energy unit '{}'".format(unit_name))
        self._unit = unit_name
        self._index = self._name_to_index[self._unit]

    def index(self):
        return self._index

    def factor_from_meV(self):
        return self._conversion_factors[0][self._index]

    def factor_to_meV(self):
        return self._conversion_factors[self._index][0]

    def from_meV(self, *args):
        return (_scale_string_or_float(x, self._conversion_factors[0][self._index]) for x in args)

    def to_meV(self, *args):
        return (_scale_string_or_float(x, self._conversion_factors[self._index][0]) for x in args)

    def factor_from(self, unit_from):
        try:
            return self._conversion_factors[self._name_to_index[unit_from]][self._index]
        except KeyError:
            raise ValueError("Unrecognised energy unit '{}'".format(unit_from))

    def factor_to(self, unit_to):
        try:
            return self._conversion_factors[self._index][self._name_to_index[unit_to]]
        except KeyError:
            raise ValueError("Unrecognised energy unit '{}'".format(unit_to))

    def convert_from(self, unit_from, *args):
        return (_scale_string_or_float(x, self.factor_from(unit_from)) for x in args)

    def convert_to(self, unit_to, *args):
        return (_scale_string_or_float(x, self.factor_to(unit_to)) for x in args)

    def label(self):
        if MPL_COMPAT:
            return 'Energy Transfer (' + self._unit + ')'
        else:
            return 'Energy Transfer (' + self._label_latex[self._unit] + ')'

    @classmethod
    def get_index(cls, unit_name):
        try:
            return cls._name_to_index[unit_name]
        except KeyError:
            raise ValueError("Unrecognised energy unit '{}'".format(unit_name))

    @classmethod
    def get_all_units(cls):
        return cls._available_units


def convert_energy_to_meV(y, energy_axis_units):
    if 'meV' not in energy_axis_units:
        return np.array(y) * EnergyUnits(energy_axis_units).factor_from_meV()
