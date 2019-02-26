"""
Module for handling units.
At present only the energy units "meV" ("DeltaE") and "cm-1" ("DeltaE_inWavenumber")
are defined, but they are separated into a module for possible future expansion.
"""

from __future__ import (absolute_import, division, print_function)
from six import string_types

class EnergyUnits(object):

    _available_units = ['meV', 'cm-1']
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
        conv = (float(x) * self._conversion_factors[0][self._index] for x in args)
        return ('{:.2f}'.format(s) for s in conv) if isinstance(args[0], string_types) else conv

    def to_meV(self, *args):
        conv = (float(x) * self._conversion_factors[self._index][0] for x in args)
        return ('{:.2f}'.format(s) for s in conv) if isinstance(args[0], string_types) else conv

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
        conv = (float(x) * self.factor_from(unit_from) for x in args)
        return ('{:.2f}'.format(s) for s in conv) if isinstance(args[0], string_types) else conv

    def convert_to(self, unit_to, *args):
        conv = (float(x) * self.factor_to(unit_to) for x in args)
        return ('{:.2f}'.format(s) for s in conv) if isinstance(args[0], string_types) else conv

    @classmethod
    def get_index(cls, unit_name):
        try:
            return cls._name_to_index[unit_name]
        except KeyError:
            raise ValueError("Unrecognised energy unit '{}'".format(unit_name))

    @classmethod
    def get_all_units(cls):
        return cls._available_units
