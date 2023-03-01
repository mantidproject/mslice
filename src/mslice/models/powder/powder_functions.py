import numpy as np

from scipy import constants

from mantid.geometry import CrystalStructure, ReflectionGenerator, ReflectionConditionFilter

from mslice.models.labels import is_momentum, is_twotheta
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.mantid.mantid_algorithms import LoadCIF

# energy to wavelength conversion E = h^2/(2*m_n*l^2)
E2L = 1.e23 * constants.h ** 2 / (2 * constants.m_n * constants.e)

crystal_structure = {'Copper': ['3.6149 3.6149 3.6149', 'F m -3 m', 'Cu 0 0 0 1.0 0.05'],
                     'Aluminium': ['4.0495 4.0495 4.0495', 'F m -3 m', 'Al 0 0 0 1.0 0.05'],
                     'Niobium': ['3.3004 3.3004 3.3004', 'I m -3 m', 'Nb 0 0 0 1.0 0.05'],
                     'Tantalum': ['3.3013 3.3013 3.3013', 'I m -3 m', 'Ta 0 0 0 1.0 0.05']}


def compute_dvalues(d_min, d_max, structure):
    generator = ReflectionGenerator(structure)
    hkls = generator.getUniqueHKLsUsingFilter(d_min, d_max,
                                              ReflectionConditionFilter.StructureFactor)
    dvalues = np.sort(np.array(generator.getDValues(hkls)))[::-1]
    return dvalues


def _compute_powder_line_momentum(ws_name, q_axis, element, cif_file):
    two_pi = 2.0 * np.pi
    d_min = two_pi / q_axis.end
    d_max = two_pi / np.max([q_axis.start, 0.01])
    structure = _crystal_structure(ws_name, element, cif_file)
    dvalues = compute_dvalues(d_min, d_max, structure)
    x = two_pi / dvalues
    return x


def _compute_powder_line_degrees(ws_name, theta_axis, element, efixed, cif_file):
    wavelength = np.sqrt(E2L / efixed)
    d_min = wavelength / (2 * np.sin(np.deg2rad(theta_axis.end * 0.5)))
    d_max = wavelength / (2 * np.sin(np.deg2rad(theta_axis.start * 0.5)))
    structure = _crystal_structure(ws_name, element, cif_file)
    dvalues = compute_dvalues(d_min, d_max, structure)
    x = 2 * np.arcsin(wavelength / 2 / dvalues) * 180 / np.pi
    return x


def compute_powder_line(ws_name, axis, element, cif_file=False):
    efixed = get_workspace_handle(ws_name).e_fixed
    if is_momentum(axis.units):
        x0 = _compute_powder_line_momentum(ws_name, axis, element, cif_file)
    elif is_twotheta(axis.units):
        x0 = _compute_powder_line_degrees(ws_name, axis, element, efixed, cif_file)
    else:
        raise RuntimeError("units of axis not recognised")
    x = sum([[xv, xv, np.nan] for xv in x0], [])
    y = sum([[efixed / 20, -efixed / 20, np.nan] for xv in x0], [])
    return x, y


def _crystal_structure(ws_name, element, cif_file):
    if cif_file:
        ws = get_workspace_handle(ws_name).raw_ws
        LoadCIF(Workspace=ws, InputFile=cif_file)
        return ws.sample().getCrystalStructure()
    else:
        return CrystalStructure(*crystal_structure[element])
