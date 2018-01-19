from __future__ import (absolute_import, division, print_function)
from mslice.models.projection.powder.projection_calculator import ProjectionCalculator
from mslice.presenters.presenter_utility import PresenterUtility
from mslice.views.powder_projection_view import PowderView
from mslice.widgets.projection.powder.command import Command
from .interfaces.powder_projection_presenter import PowderProjectionPresenterInterface
from .validation_decorators import require_main_presenter


class PowderProjectionPresenter(PresenterUtility, PowderProjectionPresenterInterface):
    def __init__(self, powder_view, projection_calculator):
        self._powder_view = powder_view
        self._projection_calculator = projection_calculator
        if not isinstance(self._powder_view,PowderView):
            raise TypeError("powder_view is not of type PowderView")
        if not isinstance(self._projection_calculator,ProjectionCalculator):
            raise TypeError("projection_calculator is not of type ProjectionCalculator")

        #Add rest of options
        self._available_axes = projection_calculator.available_axes()
        self._available_units = projection_calculator.available_units()
        self._powder_view.populate_powder_u1(self._available_axes)
        self._powder_view.populate_powder_u2(self._available_axes)
        self._powder_view.set_powder_u2(self._available_axes[-1])
        self._powder_view.populate_powder_projection_units(self._available_units)
        self._main_presenter = None

    def notify(self, command):
        self._clear_displayed_error(self._powder_view)
        self._powder_view.busy.emit(True)
        if command == Command.CalculatePowderProjection:
            self._calculate_powder_projection()
        elif command == Command.U1Changed:
            self._axis_changed(1)
        elif command == Command.U2Changed:
            self._axis_changed(2)
        else:
            raise ValueError("Powder Projection Presenter received an unrecognised command")
        self._powder_view.busy.emit(False)

    def _calculate_powder_projection(self):
        selected_workspaces = self._get_main_presenter().get_selected_workspaces()
        axis1 = self._powder_view.get_powder_u1()
        axis2 = self._powder_view.get_powder_u2()
        if axis1 == axis2:
            raise ValueError('equal axis')
        if not selected_workspaces:
            self._powder_view.display_message_box("No workspace is selected")
            return
        units = self._powder_view.get_powder_units()
        outws = []
        for workspace in selected_workspaces:
            outws.append(self._projection_calculator.calculate_projection(workspace, axis1, axis2, units))
        self._get_main_presenter().update_displayed_workspaces()
        self._get_main_presenter().set_selected_workspaces(outws)
        self._get_main_presenter().change_ws_tab(1)

    @require_main_presenter
    def _get_main_presenter(self):
        return self._main_presenter

    def _axis_changed(self, axis):
        """This is a private method which makes sure u1 and u2 are always different and one is always DeltaE"""
        curr_axis = axis - 1
        other_axis = axis % 2
        num_items = len(self._available_axes)
        axes = [self._powder_view.get_powder_u1(), self._powder_view.get_powder_u2()]
        axes_set = [self._powder_view.set_powder_u1, self._powder_view.set_powder_u2]
        name_to_index = dict((val,id) for id, val in enumerate(self._available_axes))
        if axes[curr_axis] == axes[other_axis]:
            new_index = (name_to_index[axes[other_axis]] + 1) % num_items
            axes_set[other_axis](self._available_axes[new_index])
        # Assuming DeltaE is always the last axes option.
        if axes[curr_axis] != self._available_axes[-1] and axes[other_axis] != self._available_axes[-1]:
            axes_set[other_axis](self._available_axes[-1])

    def set_workspace_provider(self, workspace_provider):
        self._projection_calculator.set_workspace_provider(workspace_provider)

    def workspace_selection_changed(self):
        workspace_selection = self._main_presenter.get_selected_workspaces()
        try:
            for workspace in workspace_selection:
                self._projection_calculator.validate_workspace(workspace)
                self._projection_calculator.get_emode(workspace)
        except TypeError as e:
            self._powder_view.disable_calculate_projections(True)
            self._powder_view.display_projection_error(str(e))
        else:
            self._powder_view.disable_calculate_projections(False)
            self._powder_view.display_projection_error("")
