from workspace import Workspace
from histogram_workspace import HistogramWorkspace
from mantid.simpleapi import BinMD


class PixelWorkspace(Workspace):
    """workspace wrapper for MDEventWorkspace. Converts to HistogramWorkspace internally."""

    def __init__(self, mantid_workspace):
        """Can be initialized with either MDEventWorkspace or HistogramWorkspace wrapper"""
        if isinstance(mantid_workspace, HistogramWorkspace):
            self._histo_ws = mantid_workspace
        else:
            Workspace.__init__(self, mantid_workspace)
            self._histo_ws = None

    def get_histo_ws(self):
        """Converts _raw_ws from MDEventWorkspace to MDHistoWorkspace using BinMD."""
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
                                    AlignedDim0=dim_values[0], alignedDim1=dim_values[1],
                                    alignedDim2=dim_values[2], alignedDim3=dim_values[3],
                                    alignedDim4=dim_values[4], alignedDim5=dim_values[5])
            self._histo_ws = HistogramWorkspace(histo_workspace)
        return self._histo_ws

    def get_signal(self):
        """Gets data values (Y axis) from the workspace as a numpy array. Overrides Workspace method."""
        return self.get_histo_ws().get_signal()

    def get_error(self):
        """Gets error values (E) from the workspace as a numpy array. Overrides Workspace method."""
        return self.get_histo_ws().get_error()

    def get_variance(self):
        """Gets variance (error^2) from the workspace as a numpy array. Overrides Workspace method"""
        return self.get_histo_ws().get_variance()

    def _binary_op_array(self, operator, other):
        """
        Perform binary operation (+,-,*,/) using a 1D numpy array.

        Overrides Workspace method.
        Note this wraps the result in HistogramWorkspace object, which is then passed
         to PixelWorkspace constructor in Workspace._binary_op.
        """
        return HistogramWorkspace(self.get_histo_ws()._binary_op_array(operator, other))

    def __pow__(self, other):
        new = self.get_histo_ws()
        while other > 1:
            new = new * self._histo_ws
            other -= 1
        return PixelWorkspace(new)
