import numpy as np
import operator as op
from mslice.util.numpy_helper import apply_with_corrected_shape, transform_array_to_workspace
from mantid.simpleapi import CloneWorkspace


class HistoMixin(object):

    def get_signal(self):
        """Gets data values (Y axis) from the workspace as a numpy array."""
        return np.squeeze(self._raw_ws.getSignalArray().copy())

    def get_error(self):
        """Gets error values (E) from the workspace as a numpy array."""
        return np.sqrt(self.get_variance(False))

    def get_variance(self, copy=True):
        """Gets variance (error^2) from the workspace as a numpy array."""
        variance = self._raw_ws.getErrorSquaredArray()
        return np.squeeze(variance.copy() if copy else variance)

    def set_signal(self, signal):
        self._raw_ws.setSignalArray(signal)

    def _binary_op_array(self, operator, other):
        """
        Perform binary operation (+,-,*,/) using a 1D numpy array.

        :param operator: binary operator to apply (add/sub/mul/div)
        :param other: 1D numpy array to use with operator.
        :return: new HistogramWorkspace
        """
        signal = self.get_signal()
        error = self.get_variance()
        new_ws = CloneWorkspace(InputWorkspace=self._raw_ws, StoreInADS=False)
        array_size_error = RuntimeError("List or array must have same number of elements as an axis of the workspace")

        new_signal = apply_with_corrected_shape(operator, signal, other, array_size_error)
        new_signal = transform_array_to_workspace(new_signal, new_ws)
        new_ws.setSignalArray(new_signal)

        if operator == op.mul or operator == op.truediv:
            new_error = apply_with_corrected_shape(operator, error, other, array_size_error)
            new_error = transform_array_to_workspace(new_error, new_ws)
            new_ws.setErrorSquaredArray(new_error)
        return new_ws
