from __future__ import (absolute_import, division, print_function)

import mslice.util.mantid.init_mantid # noqa: F401
from mslice.plotting.pyplot import *  # noqa: F401
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mslice.cli.helperfunctions import is_slice, is_cut
from ._mslice_commands import *  # noqa: F401
import mslice.plotting.pyplot as plt
from mslice.views.slice_plotter import set_colorbar_label

# This is not compatible with mslice as we use a separate
# global figure manager see _mslice_commands.Show
del show  # noqa: F821


# MSlice Matplotlib Projection
class MSliceAxes(Axes):
    name = 'mslice'

    def plot(self, *args, **kwargs):
        from mslice.cli.projection_functions import PlotCutMsliceProjection
        if is_cut(*args):
            return PlotCutMsliceProjection(self, *args, **kwargs)
        else:
            return Axes.plot(self, *args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        from mslice.cli.projection_functions import PlotSliceMsliceProjection
        if is_slice(*args):
            return PlotSliceMsliceProjection(self, *args, **kwargs)
        else:
            return Axes.pcolormesh(self, *args, **kwargs)

    def recoil_line(self, workspace_name, key, recoil, cif=None):
        get_slice_plotter_presenter().add_overplot_line(workspace_name, key, recoil, cif)
        update_overplot_checklist(key)
        update_legend()

    def intensity_plot(self, workspace_name, method_name, temp_value, temp_dependent, label):
        plt.gcf().delaxes(plt.gcf().axes[1])
        intensity_action_keys = {
            'show_scattering_function': 'action_sqe',
            'show_dynamical_susceptibility': 'action_chi_qe',
            'show_dynamical_susceptibility_magnetic': 'action_chi_qe_magnetic',
            'show_d2sigma': 'action_d2sig_dw_de',
            'show_symmetrised': 'action_symmetrised_sqe',
            'show_gdos': 'action_gdos',
        }
        plot_handler = GlobalFigureManager.get_active_figure()._plot_handler
        plot_window = GlobalFigureManager.get_active_figure().window

        intensity_method = getattr(get_slice_plotter_presenter(), method_name)
        intensity_action = getattr(plot_window, intensity_action_keys[method_name])
        intensity_action.setChecked(True)

        if temp_dependent:
            get_slice_plotter_presenter().set_sample_temperature(workspace_name, temp_value)
        ax = plot_handler.show_intensity_plot(intensity_action, intensity_method, False)
        set_colorbar_label(label)
        update_legend()

        return ax

    def change_axis_scale(self, colorbar_range, logarithmic):
        plot_handler = GlobalFigureManager.get_active_figure()._plot_handler
        plot_handler.change_axis_scale(colorbar_range, logarithmic)


register_projection(MSliceAxes)


def update_overplot_checklist(key):
    overplot_keys = {1: 'Hydrogen', 2: 'Deuterium', 4: 'Helium', 'Aluminium': 'Aluminium', 'Copper': 'Copper',
                     'Niobium': 'Niobium', 'Tantalum': 'Tantalum', 'Arbitrary Nuclei': 'Arbitrary Nuclei',
                     'CIF file': 'CIF file'}
    if key not in overplot_keys:
        plot_handler = GlobalFigureManager.get_active_figure()._plot_handler
        plot_handler._arb_nuclei_rmm = key
        key = 'Arbitrary Nuclei'

    window = GlobalFigureManager.get_active_figure().window
    getattr(window, 'action_' + overplot_keys[key].replace(' ', '_').lower()).setChecked(True)


def update_legend():
    plot_handler = GlobalFigureManager.get_active_figure()._plot_handler
    plot_handler.update_legend()
