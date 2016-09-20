from PyQt4.QtGui import QWidget,QListWidgetItem,QFileDialog, QInputDialog,QMessageBox
from PyQt4.QtCore import pyqtSignal
from command import Command
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from presenters.workspace_manager_presenter import WorkspaceManagerPresenter
from views.workspace_view import WorkspaceView
from workspacemanager_ui import Ui_Form


class WorkspaceManagerWidget(QWidget,Ui_Form,WorkspaceView):
    """A Widget that allows user to perform basic workspace save/load/rename/group/delete operations on workspaces"""

    error_occurred = pyqtSignal('QString')

    def __init__(self,parent):
        super(WorkspaceManagerWidget,self).__init__(parent)
        self.setupUi(self)
        self.btnWorkspaceSave.clicked.connect(self._btn_clicked)
        self.btnLoad.clicked.connect(self._btn_clicked)
        self.btnWorkspaceCompose.clicked.connect(self._btn_clicked)
        self.btnWorkspaceRemove.clicked.connect(self._btn_clicked)
        self.btnWorkspaceGroup.clicked.connect(self._btn_clicked)
        self.btnRename.clicked.connect(self._btn_clicked)
        self.button_mappings = {self.btnWorkspaceGroup: Command.GroupSelectedWorkSpaces,
                                self.btnWorkspaceRemove: Command.RemoveSelectedWorkspaces,
                                self.btnWorkspaceSave: Command.SaveSelectedWorkspace,
                                self.btnWorkspaceCompose: Command.ComposeWorkspace,
                                self.btnLoad: Command.LoadWorkspace,
                                self.btnRename: Command.RenameWorkspace
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
        selected_workspaces = map(lambda x: str(x.text()), self.lstWorkspaces.selectedItems())
        return list(selected_workspaces)

    def get_workspace_to_load_path(self):
        path = QFileDialog.getOpenFileName()
        return str( path )

    def get_workspace_to_save_filepath(self):
        extension = 'Nexus file (*.nxs)'
        path = QFileDialog.getSaveFileName(filter=extension)
        return str(path)

    def get_workspace_new_name(self):
        name,success = QInputDialog.getText(self,"Workspace New Name","Enter the new name for the workspace :      ")
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

    def error_unable_to_open_file(self):
        self._display_error('MSlice was not able to load the selected file')

    def confirm_overwrite_workspace(self):
        reply = QMessageBox.question(self,'Confirm Overwrite','The workspace you want to load has the same name as'
                                                              'an existing workspace, Are you sure you want to '
                                                              'overwrite it? ',QMessageBox.Yes |
                                                                              QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            return True
        else:
            return False

    def error_invalid_save_path(self):
        self._display_error('No files were saved')
        
    def no_workspace_has_been_loaded(self):
        self._display_error('No new workspaces have been loaded')

    def get_presenter(self):
        return self._presenter

    def list_item_changed(self, *args):
        self._presenter.notify(Command.SelectionChanged)

    def error_unable_to_save(self):
        self._display_error("Something went wrong while trying to save")