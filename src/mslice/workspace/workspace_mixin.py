from __future__ import (absolute_import, division, print_function)
import numpy as np
import operator as op

from mantid.simpleapi import CloneWorkspace, PowerMD
from mslice.util.numpy_helper import apply_with_corrected_shape


# Other operators are defined when MSlice is imported in _workspace_ops.attach_binary_operators()
class WorkspaceOperatorMixin(object):
    def __neg__(self):
        return self * -1

    def __pow__(self, exponent):
        return self.rewrap(PowerMD(InputWorkspace=self._raw_ws, OutputWorkspace="_",
                                   Exponent=exponent, StoreInADS=False))


class WorkspaceMixin(object):

    def get_coordinates(self):
        """
        Gets dimensions and bins of a workspace.

        :return: dict where keys are the names of the dimension and the values are a numpy array of the bins.
        """
        coords = {}
        for i in range(self._raw_ws.getNumDims()):
            dim = self._raw_ws.getDimension(i)
            coords[dim.name] = np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins())
        return coords

    def get_signal(self):
        """Gets data values (Y axis) from the workspace as a numpy array."""
        return self._raw_ws.extractY()

    def get_error(self):
        """Gets error values (E) from the workspace as a numpy array."""
        return self._raw_ws.extractE()

    def get_variance(self):
        """Gets variance (error^2) from the workspace as a numpy array."""
        return np.square(self.get_error())

    def set_signal(self, signal):
        self._set_signal_raw(self.raw_ws, signal)

    def _set_signal_raw(self, raw_ws, signal):
        for i in range(raw_ws.getNumberHistograms()):
            raw_ws.setY(i, signal[i])

    def _set_error_raw(self, raw_ws, error):
        for i in range(raw_ws.getNumberHistograms()):
            raw_ws.setE(i, error[i])

    def _binary_op_array(self, operator, other):
        """Perform binary operation using a numpy array with the same number of elements as an axis of _raw_ws signal"""
        signal = self.get_signal()
        error = self.get_error()
        new_ws = CloneWorkspace(InputWorkspace= self._raw_ws, StoreInADS=False)
        array_size_error = RuntimeError("List or array must have same number of elements as an axis of the workspace")
        new_signal = apply_with_corrected_shape(operator, signal, other, array_size_error)
        self._set_signal_raw(new_ws, new_signal)

        if operator == op.mul or operator == op.truediv:
            new_error = apply_with_corrected_shape(operator, error, other, array_size_error)
            self._set_error_raw(new_ws, new_error)
        return new_ws

    def get_saved_cut_parameters(self, axis=None):
        try:
            if axis is None:
                axis = self._cut_params['previous_axis']
            return self._cut_params[axis], axis
        except KeyError:
            return None, None

    def set_saved_cut_parameters(self, axis, params):
        self._cut_params[axis] = params
        self._cut_params['previous_axis'] = axis

    def is_axis_saved(self, axis):
        return True if axis in self._cut_params else False

    @property
    def name(self):
        return self._name

    @property
    def raw_ws(self):
        return self._raw_ws
