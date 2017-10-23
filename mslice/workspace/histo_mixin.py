from __future__ import (absolute_import, division, print_function)
import numpy as np
from .workspace_mixin import run_child_alg


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

    def _binary_op_array(self, op, other):
        """
        Perform binary operation (+,-,*,/) using a 1D numpy array.

        CreateMDHistoWorkspace using numpy array and then use ReplicateMD so it matches the shape of _raw_ws.
        Then, apply operator.

        :param operator: binary operator to apply (add/sub/mul/div)
        :param other: 1D numpy array to use with operator.
        :return: new HistogramWorkspace
        """
        min = np.amin(other)
        max = np.amax(other)
        size = other.size
        ws = run_child_alg('CreateMDHistoWorkspace', Dimensionality=1, Extents='' + str(min) + ',' + str(max),
                           SignalInput=other, ErrorInput=other, NumberOfBins=str(size),
                           Names=self._raw_ws.getDimension(0).getName(), Units='MomentumTransfer')
        try:
            replicated = run_child_alg('ReplicateMD', ShapeWorkspace=self._raw_ws, DataWorkspace=ws)
        except RuntimeError:
            raise RuntimeError("List or array must have same number of elements as an axis of the workspace")
        # return operator(self._raw_ws, replicated)
        if op.__name__ == 'add':
            alg = 'PlusMD'
        elif op.__name__ == 'sub':
            alg = 'MinusMD'
        elif op.__name__ == 'mul':
            alg = 'MultiplyMD'
        else:
            alg = 'DivideMD'
        return run_child_alg(alg, LHSWorkspace=self._raw_ws, RHSWorkspace=replicated)
