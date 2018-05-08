from __future__ import (absolute_import, division, print_function)
from six import string_types
import numpy as np

from mantid.api import MDNormalization, WorkspaceUnitValidator
from mantid.geometry import CrystalStructure, ReflectionGenerator, ReflectionConditionFilter
from scipy import constants

from .slice_algorithm import SliceAlgorithm
from mslice.models.alg_workspace_ops import AlgWorkspaceOps
from mslice.models.workspacemanager.workspace_algorithms import (propagate_properties, run_algorithm)
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle, get_workspace_name
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.workspace import Workspace

KB_MEV = constants.value('Boltzmann constant in eV/K') * 1000
E_TO_K = np.sqrt(2 * constants.neutron_mass * constants.elementary_charge / 1000) / constants.hbar
E2L = 1.e23 * constants.h**2 / (2 * constants.m_n * constants.e)  # energy to wavelength conversion E = h^2/(2*m_n*l^2)
crystal_structure = {'Copper': ['3.6149 3.6149 3.6149', 'F m -3 m', 'Cu 0 0 0 1.0 0.05'],
                     'Aluminium': ['4.0495 4.0495 4.0495', 'F m -3 m', 'Al 0 0 0 1.0 0.05'],
                     'Niobium': ['3.3004 3.3004 3.3004', 'I m -3 m', 'Nb 0 0 0 1.0 0.05'],
                     'Tantalum': ['3.3013 3.3013 3.3013', 'I m -3 m', 'Ta 0 0 0 1.0 0.05']}


class MantidSliceAlgorithm(AlgWorkspaceOps, SliceAlgorithm):

    def compute_slice(self, selected_workspace, x_axis, y_axis, norm_to_one):
        workspace = get_workspace_handle(selected_workspace)
        if workspace.is_PSD:
            plot_data = self._compute_slice_PSD(workspace, x_axis, y_axis, norm_to_one)
        else:
            plot_data = self._compute_slice_nonPSD(workspace, x_axis, y_axis, norm_to_one)
        # rot90 switches the x and y axis to to plot what user expected.
        plot_data = np.rot90(plot_data)
        boundaries = [x_axis.start, x_axis.end, y_axis.start, y_axis.end]
        if norm_to_one:
            plot_data = self._norm_to_one(plot_data)
        plot = [plot_data] + [None]*5
        return plot, boundaries

    def _compute_slice_PSD(self, workspace, x_axis, y_axis, norm_to_one):
        assert isinstance(workspace, PixelWorkspace)
        raw_ws = workspace.raw_ws
        self._fill_in_missing_input(x_axis, raw_ws)
        self._fill_in_missing_input(y_axis, raw_ws)
        n_x_bins = self._get_number_of_steps(x_axis)
        n_y_bins = self._get_number_of_steps(y_axis)
        x_dim_id = raw_ws.getDimensionIndexByName(x_axis.units)
        y_dim_id = raw_ws.getDimensionIndexByName(y_axis.units)
        x_dim = raw_ws.getDimension(x_dim_id)
        y_dim = raw_ws.getDimension(y_dim_id)
        xbinning = x_dim.getName() + "," + str(x_axis.start) + "," + str(x_axis.end) + "," + str(n_x_bins)
        ybinning = y_dim.getName() + "," + str(y_axis.start) + "," + str(y_axis.end) + "," + str(n_y_bins)
        ws_name = get_workspace_name(workspace)
        thisslice = run_algorithm('BinMD', output_name='__' + ws_name, InputWorkspace=raw_ws, AxisAligned="1",
                                  AlignedDim0=xbinning, AlignedDim1=ybinning)
        propagate_properties(workspace, thisslice)
        thisslice = thisslice.raw_ws
        # perform number of events normalization
        with np.errstate(invalid='ignore'):
            if thisslice.displayNormalization() == MDNormalization.NoNormalization:
                plot_data = np.array(thisslice.getSignalArray())
                plot_data[np.where(thisslice.getNumEventsArray() == 0)] = np.nan
            else:
                plot_data = thisslice.getSignalArray() / thisslice.getNumEventsArray()
        return plot_data

    def _compute_slice_nonPSD(self, workspace, x_axis, y_axis, norm_to_one):
        ws_name = get_workspace_name(workspace)
        axes = [x_axis, y_axis]
        if x_axis.units == 'DeltaE':
            e_axis = 0
        elif y_axis.units == 'DeltaE':
            e_axis = 1
        else:
            raise RuntimeError('Cannot calculate slices without an energy axis')
        q_axis = (e_axis + 1) % 2
        ebin = '%f, %f, %f' % (axes[e_axis].start, axes[e_axis].step, axes[e_axis].end)
        qbin = '%f, %f, %f' % (axes[q_axis].start, axes[q_axis].step, axes[q_axis].end)
        if axes[q_axis].units == '|Q|':
            thisslice = run_algorithm('SofQW3', output_name='__' + ws_name, store=False, InputWorkspace=workspace,
                                      QAxisBinning=qbin, EAxisBinning=ebin, EMode=workspace.e_mode)
        else:
            thisslice = run_algorithm('ConvertSpectrumAxis', store=False, InputWorkspace=workspace, Target='Theta')
            thisslice = run_algorithm('Rebin2D', output_name='__' + ws_name, InputWorkspace=thisslice, Axis1Binning=ebin,
                                      Axis2Binning=qbin)
        plot_data = thisslice.extractY()
        propagate_properties(workspace, thisslice)
        if e_axis == 0:
            plot_data = np.transpose(plot_data)
        return plot_data

    def axis_values(self, axis, reverse=False):
        """
        Compute a set of bins for the given axis values
        :param axis: Axis object defining axis details
        :param reverse: If true then the axis should have values in decreasing order
        :return: A new numpy array containing the axis values
        """
        if reverse:
            values = np.linspace(axis.end, axis.start, self._get_number_of_steps(axis))
        else:
            values = np.linspace(axis.start, axis.end, self._get_number_of_steps(axis))
        return values

    def compute_boltzmann_dist(self, sample_temp, delta_e):
        '''calculates exp(-E/kBT), a common factor in intensity corrections'''
        kBT = sample_temp * KB_MEV
        return np.exp(-delta_e / kBT)

    def compute_chi(self, scattering_data, sample_temp, e_axis, data_rotated):
        """
        :param scattering_data: Scattering data in axes selected by user
        :param sample_temp: The sample temperature in Kelvin
        :param e_axis: Axis object defining axis details
        :param data_rotated: If true then it is assumed that the X axis=energy otherwise
        it is assumed Y-axis=energy
        :return: The dynamic susceptibility of the data
        """
        energy_transfer = self.axis_values(e_axis, reverse=not data_rotated)
        signs = np.sign(energy_transfer)
        signs[signs == 0] = 1
        boltzmann_dist =  self.compute_boltzmann_dist(sample_temp, energy_transfer)
        chi = np.pi * (signs + (boltzmann_dist * -signs))
        if data_rotated:
            chi = chi[None, :]
        else:
            chi = chi[:, None]

        return chi * scattering_data

    def compute_chi_magnetic(self, chi):
        if chi is None:
            return None
        # 291 milibarns is the total neutron cross-section for a moment of one bohr magneton
        chi_magnetic = chi / 291
        return chi_magnetic

    def compute_d2sigma(self, scattering_data, workspace, e_axis, data_rotated):
        """
        :param scattering_data: Scattering data in axes selected by user
        :param workspace: A reference to the workspace
        :param e_axis: Axis object defining axis details
        :param data_rotated: If true then it is assumed that the X axis=energy otherwise
        it is assumed Y-axis=energy
        :return: d2sigma
        """
        Ei = get_workspace_handle(workspace).e_fixed
        if Ei is None:
            return None
        ki = np.sqrt(Ei) * E_TO_K
        energy_transfer = self.axis_values(e_axis, reverse=not data_rotated)
        kf = (np.sqrt(Ei - energy_transfer)*E_TO_K)
        if data_rotated:
            kf = kf[None, :]
        else:
            kf = kf[:, None]

        return scattering_data * kf / ki

    def compute_symmetrised(self, scattering_data, sample_temp, e_axis, data_rotated):
        energy_transfer = self.axis_values(e_axis, reverse=not data_rotated)
        negative_de = energy_transfer[energy_transfer < 0]
        negative_de_len = len(negative_de)
        boltzmann_dist = self.compute_boltzmann_dist(sample_temp, negative_de)
        if data_rotated:
            # xaxis=dE
            lhs = scattering_data[:, :negative_de_len] * boltzmann_dist
            rhs = scattering_data[:, negative_de_len:]
            return np.concatenate((lhs, rhs), 1)
        else:
            rhs = scattering_data[-negative_de_len:, :] * boltzmann_dist[:, None]
            lhs = scattering_data[:-negative_de_len, :]
            return np.concatenate((lhs, rhs))


    def compute_gdos(self, scattering_data, sample_temp, q_axis, e_axis, data_rotated):
        energy_transfer = self.axis_values(e_axis, reverse=not data_rotated)
        momentum_transfer = self.axis_values(q_axis, reverse=data_rotated)
        momentum_transfer = np.square(momentum_transfer, out=momentum_transfer)
        boltzmann_dist = self.compute_boltzmann_dist(sample_temp, energy_transfer)
        if data_rotated:
            gdos = scattering_data / momentum_transfer[:,None]
            gdos *= energy_transfer
            gdos *= (1 - boltzmann_dist)[None, :]
        else:
            gdos = scattering_data / momentum_transfer
            gdos *= energy_transfer[:, None]
            gdos *= (1 - boltzmann_dist)[:, None]

        return gdos

    def sample_temperature(self, ws_name, sample_temp_fields):
        ws = get_workspace_handle(ws_name).raw_ws
        sample_temp = None
        for field_name in sample_temp_fields:
            try:
                sample_temp = ws.run().getLogData(field_name).value
            except RuntimeError:
                pass
            except AttributeError:
                sample_temp = ws.getExperimentInfo(0).run().getLogData(field_name).value
        try:
            float(sample_temp)
        except (ValueError, TypeError):
            pass
        else:
            return sample_temp
        if isinstance(sample_temp, string_types):
            sample_temp = self.get_sample_temperature_from_string(sample_temp)
        if isinstance(sample_temp, np.ndarray) or isinstance(sample_temp, list):
            sample_temp = np.mean(sample_temp)
        return sample_temp

    def get_sample_temperature_from_string(self, string):
        pos_k = string.find('K')
        if pos_k == -1:
            return None
        k_string = string[pos_k - 3:pos_k]
        sample_temp = float(''.join(c for c in k_string if c.isdigit()))
        return sample_temp

    def compute_recoil_line(self, ws_name, axis, relative_mass=1):
        efixed = get_workspace_handle(ws_name).e_fixed
        x_axis = np.arange(axis.start, axis.end, axis.step)
        if axis.units == 'MomentumTransfer' or axis.units == '|Q|':
            momentum_transfer = x_axis
            line = np.square(momentum_transfer * 1.e10 * constants.hbar) / (2 * relative_mass * constants.neutron_mass) /\
                (constants.elementary_charge / 1000)
        elif axis.units == 'Degrees':
            tth = x_axis * np.pi / 180.
            if 'Direct' in get_workspace_handle(ws_name).e_mode:
                line = efixed * (2 - 2 * np.cos(tth)) / (relative_mass + 1 - np.cos(tth))
            else:
                line = efixed * (2 - 2 * np.cos(tth)) / (relative_mass - 1 + np.cos(tth))
        else:
            raise RuntimeError("units of axis not recognised")
        return x_axis, line

    def compute_powder_line(self, ws_name, axis, element, cif_file=False):
        efixed = get_workspace_handle(ws_name).e_fixed
        if axis.units == 'MomentumTransfer' or axis.units == '|Q|':
            x0 = self._compute_powder_line_momentum(ws_name, axis, element, cif_file)
        elif axis.units == 'Degrees':
            x0 = self._compute_powder_line_degrees(ws_name, axis, element, efixed, cif_file)
        else:
            raise RuntimeError("units of axis not recognised")
        x = sum([[xv, xv, np.nan] for xv in x0], [])
        y = sum([[efixed / 20,  -efixed / 20, np.nan] for xv in x0], [])
        return x, y

    def _compute_powder_line_momentum(self, ws_name, q_axis, element, cif_file):
        d_min = (2 * np.pi) / q_axis.end
        d_max = (2 * np.pi) / q_axis.start
        structure = self._crystal_structure(ws_name, element, cif_file)
        dvalues = self.compute_dvalues(d_min, d_max, structure)
        x = (2 * np.pi) / dvalues
        return x

    def _crystal_structure(self, ws_name, element, cif_file):
        if cif_file:
            ws = get_workspace_handle(ws_name).raw_ws
            run_algorithm('LoadCIF', store=False, InputWorkspace=ws, InputFile=cif_file)
            return ws.sample().getCrystalStructure()
        else:
            return CrystalStructure(crystal_structure[element][0], crystal_structure[element][1],
                                    crystal_structure[element][2])

    def _compute_powder_line_degrees(self, ws_name, theta_axis, element, efixed, cif_file):
        wavelength = np.sqrt(E2L / efixed)
        d_min = wavelength / (2 * np.sin(np.deg2rad(theta_axis.end / 2)))
        d_max = wavelength / (2 * np.sin(np.deg2rad(theta_axis.start / 2)))
        structure = self._crystal_structure(ws_name, element, cif_file)
        dvalues = self.compute_dvalues(d_min, d_max, structure)
        x = 2 * np.arcsin(wavelength / 2 / dvalues) * 180 / np.pi
        return x

    def compute_dvalues(self, d_min, d_max, structure):
        generator = ReflectionGenerator(structure)
        hkls = generator.getUniqueHKLsUsingFilter(d_min, d_max, ReflectionConditionFilter.StructureFactor)
        dvalues = np.sort(np.array(generator.getDValues(hkls)))[::-1]
        return dvalues

    def _norm_to_one(self, data):
        data_range = np.nanmax(data) - np.nanmin(data)
        return (data - np.nanmin(data))/data_range

    def is_sliceable(self, workspace):
        ws = get_workspace_handle(workspace)
        if isinstance(ws, PixelWorkspace):
            return True
        else:
            validator = WorkspaceUnitValidator('DeltaE')
            return isinstance(ws, Workspace) and validator.isValid(ws.raw_ws) == ''
