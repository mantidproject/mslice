from __future__ import (absolute_import, division, print_function)
from .histogram_workspace import HistogramWorkspace
from mantid.simpleapi import BinMD


class PixelMixin(object):

    def get_histo_ws(self):
        """Converts _raw_ws from MDEventWorkspace to MDHistoWorkspace using BinMD, and caches result in _histo_ws"""
        if self._histo_ws is None:
            dim_values = []
            for x in range(6):
                try:
                    dim = self._raw_ws.getDimension(x)
                    dim_info = dim.getName() + ',' + str(dim.getMinimum()) + ',' + str(dim.getMaximum()) + ',' + str(100)
                except RuntimeError:
                    dim_info = None
                dim_values.append(dim_info)
            histo_workspace = BinMD(InputWorkspace=self._raw_ws, OutputWorkspace=str(self),
                                    AlignedDim0=dim_values[0], AlignedDim1=dim_values[1],
                                    AlignedDim2=dim_values[2], AlignedDim3=dim_values[3],
                                    AlignedDim4=dim_values[4], AlignedDim5=dim_values[5])
            self._histo_ws = HistogramWorkspace(histo_workspace)
        return self._histo_ws

    def get_signal(self):
        """Gets data values (Y axis) from the workspace as a numpy array."""
        return self.get_histo_ws().get_signal()

    def get_error(self):
        """Gets error values (E) from the workspace as a numpy array."""
        return self.get_histo_ws().get_error()

    def get_variance(self):
        """Gets variance (error^2) from the workspace as a numpy array."""
        return self.get_histo_ws().get_variance()

    def _binary_op_array(self, operator, other):
        """
        Perform binary operation (+,-,*,/) using a 1D numpy array.

        Note this wraps the result in HistogramWorkspace object, which is then passed
         to PixelWorkspace constructor in Workspace._binary_op.
        """
        return HistogramWorkspace(self.get_histo_ws()._binary_op_array(operator, other))

    def __pow__(self, other):
        return self.rewrap(self.get_histo_ws().__pow__(other))
