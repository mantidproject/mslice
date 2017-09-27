import numpy as np
from mantid.simpleapi import CreateMDHistoWorkspace, ReplicateMD
from mantid.api import AlgorithmManager


class HistoMixin(object):

    def get_signal(self):
        """Gets data values (Y axis) from the workspace as a numpy array. Overrides Workspace method."""
        return self._raw_ws.getSignalArray()

    def get_error(self):
        """Gets error values (E) from the workspace as a numpy array. Overrides Workspace method."""
        return np.sqrt(self.get_variance())

    def get_variance(self):
        """Gets variance (error^2) from the workspace as a numpy array. Overrides Workspace method"""
        return self._raw_ws.getErrorSquaredArray()

    def _binary_op_array(self, operator, other):
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
        ws = CreateMDHistoWorkspace(Dimensionality=1, Extents='' + str(min) + ',' + str(max),
                                    SignalInput=other, ErrorInput=other, NumberOfBins=str(size),
                                    Names=self._raw_ws.getDimension(0).getName(), Units='MomentumTransfer')
        try:
            replicated = ReplicateMD(self._raw_ws, ws)
            return operator(self._raw_ws, replicated)
        except RuntimeError:
            raise RuntimeError("List or array must have same number of elements as an axis of the workspace")

    def __pow__(self, exponent):
        pow_alg = AlgorithmManager.createUnmanaged('PowerMD')
        pow_alg.setChild(True)
        pow_alg.initialize()
        pow_alg.setProperty('InputWorkspace', self._raw_ws)
        pow_alg.setProperty('OutputWorkspace', 'dummy')
        pow_alg.setProperty('Exponent', exponent)
        pow_alg.execute()
        return self.rewrap(pow_alg.getProperty('OutputWorkspace').value)
