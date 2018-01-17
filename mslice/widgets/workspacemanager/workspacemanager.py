from __future__ import (absolute_import, division, print_function)

from mslice.util.qt.QtCore import Signal
from mslice.util.qt.QtWidgets import QWidget, QListWidgetItem, QFileDialog, QInputDialog, QMessageBox

from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from mslice.presenters.workspace_manager_presenter import WorkspaceManagerPresenter
from mslice.util.qt import load_ui
from mslice.views.workspace_view import WorkspaceView
from .command import Command


class WorkspaceManagerWidget(WorkspaceView, QWidget):
    """A Widget that allows user to perform basic workspace save/load/rename/delete operations on workspaces"""

    error_occurred = Signal('QString')
    busy = Signal(bool)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        load_ui(__file__, 'workspacemanager.ui', self)
        self.button_mappings = {}
        self._main_window = None
        self.lstWorkspaces2D.itemSelectionChanged.connect(self.list_item_changed)
        self._presenter = WorkspaceManagerPresenter(self, MantidWorkspaceProvider())

    def _display_error(self, error_string):
        self.error_occurred.emit(error_string)

    def _btn_clicked(self):
        sender = self.sender()
        try:
            command = self.button_mappings[sender]
        except KeyError:
            raise Exception('Invalid sender')
        self._presenter.notify(command)

    def display_loaded_workspaces(self, workspaces):
        onscreen_workspaces = []
        for index in range(self.lstWorkspaces2D.count()):
            qitem = self.lstWorkspaces2D.item(index)
            onscreen_workspaces.append(str(qitem.text()))
        for workspace in workspaces:
            if workspace in onscreen_workspaces:
                onscreen_workspaces.remove(workspace)
                continue
            item = QListWidgetItem(workspace)
            self.lstWorkspaces2D.addItem(item)

        # remove any onscreen workspaces that are no longer here
        items = [] #items contains (qlistitem, index) tuples
        for index in range(self.lstWorkspaces2D.count()):
            items.append(self.lstWorkspaces2D.item(index))
        for item in items:
            if str(item.text()) in onscreen_workspaces:
                self.remove_item_from_list(item)

    def remove_item_from_list(self,qlistwidget_item):
        """Remove given qlistwidget_item from list.

        Must be done in seperate function because items are removed by index and removing an items may alter the indexes
        of other items"""
        text = qlistwidget_item.text()
        for index in range(self.lstWorkspaces2D.count()):
            if self.lstWorkspaces2D.item(index).text() == text:
                self.lstWorkspaces2D.takeItem(index)
                return

    def get_workspace_selected(self):
        selected_workspaces = [str(x.text()) for x in self.lstWorkspaces2D.selectedItems()]
        return list(selected_workspaces)

    def set_workspace_selected(self, index):
        for item_index in range(self.lstWorkspaces2D.count()):
            self.lstWorkspaces2D.setItemSelected(self.lstWorkspaces2D.item(item_index), False)
        for this_index in (index if hasattr(index, "__iter__") else [index]):
            self.lstWorkspaces2D.setItemSelected(self.lstWorkspaces2D.item(this_index), True)

    def get_workspace_index(self, ws_name):
        for index in range(self.lstWorkspaces2D.count()):
            if str(self.lstWorkspaces2D.item(index).text()) == ws_name:
                return index
        return -1

    def get_workspace_to_load_path(self):
        paths = QFileDialog.getOpenFileNames()
        return [str(filename) for filename in paths]

    def get_directory_to_save_workspaces(self):
        return QFileDialog.getExistingDirectory()

    def get_workspace_new_name(self):
        name, success = QInputDialog.getText(self,"Workspace New Name","Enter the new name for the workspace :      ")
        # The message above was padded with spaces to allow the whole title to show up
        if not success:
            raise ValueError('No Valid Name supplied')
        return str(name)

    def error_select_only_one_workspace(self):
        self._display_error('Please select only one workspace and then try again')

    def error_select_one_or_more_workspaces(self):
        self._display_error('Please select one or more workspaces the try again')

    def error_select_one_workspace(self):
        self._display_error('Please select a workspace then try again')

    def error_select_more_than_one_workspaces(self):
        self._display_error('Please select more than one projected workspaces then try again')

    def error_invalid_save_path(self):
        self._display_error('No files were saved')

    def get_presenter(self):
        return self._presenter

    def list_item_changed(self):
        self._presenter.notify(Command.SelectionChanged)

    def error_unable_to_save(self):
        self._display_error("Something went wrong while trying to save")

    def clear_displayed_error(self):
        self._display_error("")
