from workspacemanager_ui import Ui_Form
from workspace_view import WorkspaceView
from PyQt4.QtGui import QWidget,QListWidgetItem,QFileDialog
from workspace_manager_presenter import WorkspaceManagerPresenter
from mantid_workspace_provider import MantidWorkspaceProvider
from command import Command

class WorkspaceManagerWidget(QWidget,Ui_Form,WorkspaceView):
    def __init__(self,*args, **kwargs):
        super(WorkspaceManagerWidget,self).__init__(*args, **kwargs)
        self._presenter = WorkspaceManagerPresenter(self,MantidWorkspaceProvider())
        self.setupUi(self)
        self.btnWorkspaceSave.clicked.connect(self.btn_clicked)
        self.btnLoad.clicked.connect(self.btn_clicked)
        self.btnWorkspaceCompose.clicked.connect(self.btn_clicked)
        self.btnWorkspaceRemove.clicked.connect(self.btn_clicked)
        self.btnWorkspaceGroup.clicked.connect(self.btn_clicked)
        self.button_mappings = {self.btnWorkspaceGroup: Command.GroupSelectedWorkSpaces,
                                self.btnWorkspaceRemove: Command.RemoveSelectedWorkspaces,
                                self.btnWorkspaceSave: Command.SaveSelectedWorkspace,
                                self.btnWorkspaceCompose: Command.ComposeWorkspace,
                                self.btnLoad: Command.LoadWorkspace
                                }

    def btn_clicked(self):
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
        print 'on',onscreen_workspaces
        for workspace in workspaces:
            if workspace in onscreen_workspaces:
                onscreen_workspaces.remove(workspace)
                continue
            item = QListWidgetItem(workspace)
            item.setCheckState(0)
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
        selected_workspaces = []
        for index in range(self.lstWorkspaces.count()):
            item = self.lstWorkspaces.item(index)
            if item.checkState():
                selected_workspaces.append(str(item.text()))
        print selected_workspaces
        return selected_workspaces

    def get_workspace_to_load_path(self):
        return str(QFileDialog.getOpenFileName())

    def get_workspace_to_save_filepath(self):
        path = QFileDialog.getSaveFileName()
        if not path:
            raise ValueError('Please Select a valid path to save to ')
        return str(path)

