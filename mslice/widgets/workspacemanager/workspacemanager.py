from __future__ import (absolute_import, division, print_function)

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget, QListWidgetItem, QFileDialog, QInputDialog, QMessageBox

from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from mslice.presenters.workspace_manager_presenter import WorkspaceManagerPresenter
from mslice.util.qt import load_ui
from mslice.views.workspace_view import WorkspaceView
from .command import Command
from .inputdialog import EfInputDialog


class WorkspaceManagerWidget(QWidget, WorkspaceView):
    """A Widget that allows user to perform basic workspace save/load/rename/delete operations on workspaces"""

    error_occurred = Signal('QString')
    busy = Signal(bool)

    def __init__(self,parent):
        super(WorkspaceManagerWidget,self).__init__(parent)
        load_ui(__file__, 'workspacemanager.ui', self)
        self.btnWorkspaceSave.clicked.connect(self._btn_clicked)
        self.btnLoad.clicked.connect(self._btn_clicked)
        self.btnWorkspaceCompose.clicked.connect(self._btn_clicked)
        self.btnWorkspaceRemove.clicked.connect(self._btn_clicked)
        self.btnRename.clicked.connect(self._btn_clicked)
        self.btnCombine.clicked.connect(self._btn_clicked)
        self.button_mappings = {self.btnWorkspaceRemove: Command.RemoveSelectedWorkspaces,
                                self.btnWorkspaceSave: Command.SaveSelectedWorkspace,
                                self.btnWorkspaceCompose: Command.ComposeWorkspace,
                                self.btnLoad: Command.LoadWorkspace,
                                self.btnRename: Command.RenameWorkspace,
                                self.btnCombine: Command.CombineWorkspace
                                }
        self._main_window = None
        self.lstWorkspaces.itemSelectionChanged.connect(self.list_item_changed)
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
        for index in range(self.lstWorkspaces.count()):
            qitem = self.lstWorkspaces.item(index)
            onscreen_workspaces.append(str(qitem.text()))
        for workspace in workspaces:
            if workspace in onscreen_workspaces:
                onscreen_workspaces.remove(workspace)
                continue
            item = QListWidgetItem(workspace)
            self.lstWorkspaces.addItem(item)

        # remove any onscreen workspaces that are no longer here
        items = [] #items contains (qlistitem, index) tuples
        for index in range(self.lstWorkspaces.count()):
            items.append(self.lstWorkspaces.item(index))
        for item in items:
            if str(item.text()) in onscreen_workspaces:
                self.remove_item_from_list(item)

    def remove_item_from_list(self,qlistwidget_item):
        """Remove given qlistwidget_item from list.

        Must be done in seperate function because items are removed by index and removing an items may alter the indexes
        of other items"""
        text = qlistwidget_item.text()
        for index in range(self.lstWorkspaces.count()):
            if self.lstWorkspaces.item(index).text() == text:
                self.lstWorkspaces.takeItem(index)
                return

    def get_workspace_selected(self):
        selected_workspaces = [str(x.text()) for x in self.lstWorkspaces.selectedItems()]
        return list(selected_workspaces)

    def set_workspace_selected(self, index):
        for item_index in range(self.lstWorkspaces.count()):
            self.lstWorkspaces.setItemSelected(self.lstWorkspaces.item(item_index), False)
        for this_index in (index if hasattr(index, "__iter__") else [index]):
            self.lstWorkspaces.setItemSelected(self.lstWorkspaces.item(this_index), True)

    def get_workspace_index(self, ws_name):
        for index in range(self.lstWorkspaces.count()):
            if str(self.lstWorkspaces.item(index).text()) == ws_name:
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

    def get_workspace_efixed(self, workspace, hasMultipleWS=False):
        Ef, applyToAll, success = EfInputDialog.getEf(workspace, hasMultipleWS, None)
        if not success:
            raise ValueError('Fixed final energy not given')
        return Ef, applyToAll

    def error_select_only_one_workspace(self):
        self._display_error('Please select only one workspace and then try again')

    def error_select_one_or_more_workspaces(self):
        self._display_error('Please select one or more workspaces the try again')

    def error_select_one_workspace(self):
        self._display_error('Please select a workspace then try again')

    def error_select_more_than_one_workspaces(self):
        self._display_error('Please select more than one projected workspaces then try again')

    def error_unable_to_open_file(self, filename=None):
        self._display_error('MSlice was not able to load %s' % ('the selected file' if filename is None else filename))

    def confirm_overwrite_workspace(self):
        reply = QMessageBox.question(self,'Confirm Overwrite', 'The workspace you want to load has the same name as'
                                                               'an existing workspace, Are you sure you want to '
                                                               'overwrite it? ',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            return True
        else:
            return False

    def error_invalid_save_path(self):
        self._display_error('No files were saved')

    def no_workspace_has_been_loaded(self, filename=None):
        if filename is None:
            self._display_error('No new workspaces have been loaded')
        else:
            self._display_error('File %s has not been loaded' % (filename))

    def get_presenter(self):
        return self._presenter

    def list_item_changed(self):
        self._presenter.notify(Command.SelectionChanged)

    def error_unable_to_save(self):
        self._display_error("Something went wrong while trying to save")

    def clear_displayed_error(self):
        self._display_error("")
