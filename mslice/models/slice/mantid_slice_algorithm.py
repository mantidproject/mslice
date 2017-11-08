from __future__ import (absolute_import, division, print_function)
import numpy as np

from mantid.simpleapi import BinMD
from mantid.api import IMDEventWorkspace
from scipy import constants

from .slice_algorithm import SliceAlgorithm
from mslice.models.alg_workspace_ops import AlgWorkspaceOps
from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider

BOLTZMANN = 0.086173303  # meV/K


class MantidSliceAlgorithm(AlgWorkspaceOps, SliceAlgorithm):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def compute_slice(self, selected_workspace, x_axis, y_axis, smoothing, norm_to_one, sample_temp):
        workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        assert isinstance(workspace,IMDEventWorkspace)
        self._fill_in_missing_input(x_axis, workspace)
        self._fill_in_missing_input(y_axis, workspace)
        n_x_bins = self._get_number_of_steps(x_axis)
        n_y_bins = self._get_number_of_steps(y_axis)
        x_dim_id = workspace.getDimensionIndexByName(x_axis.units)
        y_dim_id = workspace.getDimensionIndexByName(y_axis.units)
        x_dim = workspace.getDimension(x_dim_id)
        y_dim = workspace.getDimension(y_dim_id)
        xbinning = x_dim.getName() + "," + str(x_axis.start) + "," + str(x_axis.end) + "," + str(n_x_bins)
        ybinning = y_dim.getName() + "," + str(y_axis.start) + "," + str(y_axis.end) + "," + str(n_y_bins)
        thisslice = BinMD(InputWorkspace=workspace, AxisAligned="1", AlignedDim0=xbinning, AlignedDim1=ybinning)
        # perform number of events normalization then mask cells where no data was found
        with np.errstate(invalid='ignore'):
            plot_data = thisslice.getSignalArray() / thisslice.getNumEventsArray()
        # rot90 switches the x and y axis to to plot what user expected.
        plot_data = np.rot90(plot_data)
        self._workspace_provider.delete_workspace(thisslice)
        boundaries = [x_axis.start, x_axis.end, y_axis.start, y_axis.end]
        if norm_to_one:
            plot_data = self._norm_to_one(plot_data)
        plot = [None, None, None, None]
        plot[0] = plot_data
        plot[1] = self.compute_chi(plot_data, sample_temp, y_axis)
        plot[2] = self.compute_chi_magnetic(plot[1])
        plot[3] = self.compute_d2sigma(plot[0], selected_workspace, y_axis)
        return plot, boundaries

    def compute_chi(self, scattering_data, sample_temp, y_axis):
        if sample_temp is None:
            return None
        kBT = sample_temp * BOLTZMANN
        energy_transfer = np.arange(y_axis.start, y_axis.end, y_axis.step)
        signs = np.sign(energy_transfer)
        boltzmann_dist = np.exp(-energy_transfer / kBT)
        chi = (signs + (boltzmann_dist * -signs))[:,None]
        chi = np.pi * chi * scattering_data
        return chi

    def compute_chi_magnetic(self, chi):
        if chi is None:
            return None
        # 291 milibarns is the total neutron cross-section for a moment of one bohr magneton
        chi_magnetic = chi / 291
        return chi_magnetic

    def compute_d2sigma(self, scattering_data, workspace, y_axis):
        Ei = self._workspace_provider.get_EFixed(self._workspace_provider.get_workspace_handle(workspace))
        if Ei is None:
            Ei = self._workspace_provider.get_EFixed(self._workspace_provider.get_workspace_handle(workspace[:-3]))
            if Ei is None:
                return None
        hbar = constants.value('Planck constant over 2 pi in eV s') * 1000  # convert to meV s
        ki = np.sqrt(Ei*2*constants.neutron_mass) / hbar
        energy_transfer = np.arange(y_axis.start, y_axis.end, y_axis.step)
        kf = np.sqrt(((Ei - energy_transfer)*2*constants.neutron_mass) / hbar)[:,None]
        d2sigma = scattering_data * kf / ki
        return d2sigma

    def _norm_to_one(self, data):
        data_range = data.max() - data.min()
        return (data - data.min())/data_range
