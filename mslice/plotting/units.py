"""
This module defines two Matplotlib unit classes for the energy and momentum transfer axes,
which will allow on-the-fly unit conversion even on existing plots
"""
from __future__ import (absolute_import, unicode_literals)
from matplotlib.cbook import iterable
from matplotlib import units, ticker
from scipy import constants
from mslice.util import MPL_COMPAT
from numpy import pi


class EnergyTransferUnits(object):

    # We make meV the default units, and assume that all input workspaces are using this
    # This is ensured by the loader class verifying that input are in 'DeltaE' (in Mantid
    # notation) or automatically converting to 'DeltaE' (eg. from 'DeltaE_inWavenumbers')
    _meV = constants.elementary_charge / 1000
    _conversion_factors = {None: 1,
                           'meV': 1, 
                           'cm': (constants.h * constants.c * 100) / _meV,
                           'K': constants.k / _meV,
                           'THz': (constants.h * 1e12) / _meV}
    _labels = {'meV': '(meV)',
               'cm': '(cm{})'.format('$^{-1}$' if MPL_COMPAT else '-1'),
               'K': '(K)',
               'THz': '(THz)'}

    def __init__(self, val, unit='meV'):
        self.unit = unit
        self._val = val * self._conversion_factors[unit]

    def value(self, unit):
        try:
            return self._val / self._conversion_factors[unit]
        except KeyError:
            raise ValueError('Invalid energy transfer unit {}'.format(value))

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        if value not in self._conversion_factors.keys():
            raise ValueError('Invalid energy transfer unit {}'.format(value))
        self._unit = value

    @classmethod
    def label(cls, unit):
        try:
            return 'Energy Transfer ' + cls._labels[unit]
        except KeyError:
            raise ValueError('Invalid energy transfer unit {}'.format(value))


class EnergyTransferConverter(units.ConversionInterface):

    @staticmethod
    def convert(obj, unit, axis):
        'Converts an energy transfer object to scalar or array'
        if units.ConversionInterface.is_numlike(obj):
            return obj
        # value should be an EnergyTransferUnits object or sequence of such
        if iterable(obj):
            return [en.value(unit) for en in obj]
        else:
            return obj.value(unit)

    @staticmethod
    def axisinfo(unit, axis):
        'Returns an energy transfer axis information object'
        try:
            return units.AxisInfo(label=EnergyTransferUnits.label(unit))
        except ValueError:
            return None

    @staticmethod
    def default_units(x, axis):
        'Returns the default energy transfer units or None'
        return 'meV'
 

class MomentumTransferUnits(object):

    # The default unit is 'Q', but the alternative unit (d-spacing) is not just a linear
    # scaling factor, so we return a lambda function instead.
    _forward_conversion_function = {None: lambda x: x,
                                    'Q': lambda x: x,
                                    'dSpacing': lambda x: (2*pi / x)}
    _reverse_conversion_function = {None: lambda x: x,
                                    'Q': lambda x: x,
                                    'dSpacing': lambda x: (x / 2*pi)}
    _labels = {'Q': '|Q| ({})'.format('$\mathrm{\AA}^{-1}$' if MPL_COMPAT else 'recip. Ang.'),
               'dSpacing': 'd-spacing ({})'.format('$\mathrm{\AA}' if MPL_COMPAT else 'Angstrom')}

    _alias = {None: None, 
             'MomentumTransfer': 'Q',
             '|Q|': 'Q',
             'Q': 'Q',
             'd': 'dSpacing',
             'dSpacing': 'dSpacing'}

    def __init__(self, val, unit='Q'):
        self.unit = unit
        self._val = self._forward_conversion_function[self._alias[unit]](val)

    def value(self, unit):
        try:
            return self._reverse_conversion_function[self._alias[unit]](val)
        except KeyError:
            raise ValueError('Invalid momentum transfer unit {}'.format(value))

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        if value not in self._alias.keys():
            raise ValueError('Invalid momentum transfer unit {}'.format(value))
        self._unit = value

    @classmethod
    def label(cls, unit):
        try:
            return cls._labels[cls._alias[unit]]
        except KeyError:
            raise ValueError('Invalid energy transfer unit {}'.format(value))


class MomentumTransferConverter(units.ConversionInterface):

    @staticmethod
    def convert(obj, unit, axis):
        'Converts a momentum transfer object to scalar or array'
        if units.ConversionInterface.is_numlike(obj):
            return obj
        # value should be a MomentumTransferUnits object or sequence of such
        if iterable(obj):
            return [q.value(unit) for q in obj]
        else:
            return obj.value(unit)

    @staticmethod
    def axisinfo(unit, axis):
        'Returns a momentum transfer axis information object'
        try:
            return units.AxisInfo(label=MomentumTransferUnits.label(unit))
        except ValueError:
            return None

    @staticmethod
    def default_units(x, axis):
        'Returns the default energy transfer units or None'
        return 'Q'
 

