from __future__ import (absolute_import, division, print_function)
import numpy as np
import operator
from mantid.simpleapi import CloneWorkspace, PowerMD
from mslice.util.numpy_helper import apply_with_corrected_shape


class WorkspaceMixin(object):

    def get_coordinates(self):
        """
        Gets dimensions and bins of a workspace.

        :return: dict where keys are the names of the dimension and the values are a numpy array of the bins.
        """
        coords = {}
        for i in range(self._raw_ws.getNumDims()):
            dim = self._raw_ws.getDimension(i)
            coords[dim.getName()] = np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins())
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

    def _binary_op(self, operator, other):
        """
        Delegate binary operations (+,-,*,/) performed on a workspace depending on type.

        Unwrap mantid workspace (_raw_ws), perform operation, then rewrap.

        Behaviour depends on type of other:
        List - convert to numpy array, then pass to _binary_op_array method.
        Numpy array - pass to _binary_op_array method.
        Workspace wrapper (of the same type and dimensionality) - unwrap, apply operator to each bin, rewrap.
        Else - try to apply operator to self._raw_ws. Works for numbers, unwrapped workspaces...

        :param operator: binary operator (add, sub, mul, div)
        :param other: object to add/sub/mul/div with self - can be list, numpy array, workspace, number...
        :return: new workspace wrapper with same type as self.
        """
        if isinstance(other, list):
            other = np.asarray(other)
        if isinstance(other, self.__class__):
            if self.check_dimensions(other):
                inner_res = operator(self._raw_ws, other._raw_ws)
            else:
                raise RuntimeError("workspaces must have same dimensionality for binary operations (+, -, *, /)")
        elif isinstance(other, np.ndarray):
            inner_res = self._binary_op_array(operator, other)
        else:
            inner_res = operator(self._raw_ws, other)
        return self.rewrap(inner_res)

    def _binary_op_array(self, operator, other):
        """Perform binary operation using a numpy array with the same number of elements as an axis of _raw_ws signal"""
        signal = self.get_signal()
        new_ws = CloneWorkspace(InputWorkspace= self._raw_ws, StoreInADS=False)
        error = RuntimeError("List or array must have same number of elements as an axis of the workspace")
        new_signal = apply_with_corrected_shape(operator, signal, other, error)
        self._set_signal_raw(new_ws, new_signal)
        # scale errors?
        return new_ws


    def check_dimensions(self, workspace_to_check):
        """check if a workspace has the same number of bins as self for each dimension"""
        for i in range(self._raw_ws.getNumDims()):
            if self._raw_ws.getDimension(i).getNBins() != workspace_to_check._raw_ws.getDimension(i).getNBins():
                return False
        return True

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
    def raw_ws(self):
        return self._raw_ws


    def __add__(self, other):
        return self._binary_op(operator.add, other)

    def __sub__(self, other):
        return self._binary_op(operator.sub, other)

    def __mul__(self, other):
        return self._binary_op(operator.mul, other)

    def __truediv__(self, other):
        return self._binary_op(operator.truediv, other)

    def __neg__(self):
        return self * -1

    def __pow__(self, exponent):
        return self.rewrap(PowerMD(InputWorkspace=self._raw_ws, OutputWorkspace="_", Exponent=exponent, StoreInADS=False))
