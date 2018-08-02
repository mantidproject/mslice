from __future__ import (absolute_import, division, print_function)
import numpy as np
from .workspace_mixin import run_child_alg
from mantid.simpleapi import CreateMDHistoWorkspace, ReplicateMD


class HistoMixin(object):

    def get_signal(self):
        """Gets data values (Y axis) from the workspace as a numpy array."""
        return self._raw_ws.getSignalArray().copy()

    def get_error(self):
        """Gets error values (E) from the workspace as a numpy array."""
        return np.sqrt(self.get_variance(False))

    def get_variance(self, copy=True):
        """Gets variance (error^2) from the workspace as a numpy array."""
        variance = self._raw_ws.getErrorSquaredArray()
        return variance.copy() if copy else variance

    def _binary_op_array(self, operator, other):
        """
        Perform binary operation (+,-,*,/) using a 1D numpy array.

        :param operator: binary operator to apply (add/sub/mul/div)
        :param other: 1D numpy array to use with operator.
        :return: new HistogramWorkspace
        """
        signal = self.get_signal()
        new_ws = run_child_alg('CloneWorkspace', InputWorkspace=self._raw_ws, OutputWorkspace='dummy')
        if other.size == signal.shape[1]:
            new_signal = operator(signal, other)
        elif other.size == signal.shape[0]:
            new_signal = np.transpose(operator(np.transpose(signal), other))
        else:
            raise ValueError("List or array must have same number of elements as an axis of the workspace")
        new_ws.setSignalArray(new_signal)
        return new_ws
