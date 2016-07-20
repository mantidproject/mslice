from PyQt4 import QtGui

from MSliceMainView import MainView
from workspacemanager.WorkspaceManagerPresenter import MainViewPresenter
from command import Command
from ui_Mslice import Ui_MainWindow

class MSliceGui(MainView,QtGui.QMainWindow,Ui_MainWindow):
    #TODO Should i set a metaclass?
    def __init__(self):
        super(QtGui.QMainWindow,self).__init__()
        self.setupUi(self)
        #Create a presenter for the GUI
        self.presenter = MainViewPresenter(self)

        #Setup button signals
        self.button_commands = {
            self.btnWorkspaceSave:Command.SaveSelectedWorkspaces,
            self.btnLoad:Command.LoadWorkspace, #TODO Fix this name
            self.btnWorkspaceRemove:Command.RemoveSelectedWorkspaces,
            self.btnWorkspaceGroup:Command.GroupSelectedWorkSpaces,
            self.btnWorkspaceCompose:Command.ComposeWorkspace,
            self.btnPowderCalculateProjection:Command.CalculatePowderProjection,
            self.btnSingleCalculateProjections:Command.CalculateSingleCrystalProjection,
            self.btnSingleCheckWorkspace:Command.CheckSingleCrystalWorkspace,
            self.btnSingleSaveParams:Command.SaveSingleCrystalParams,
            self.btnSingleLoadParams:Command.LoadSingleCrystalParams,
            self.btnSingleDefaultParams:Command.DefaultSingleCrystalParams,
            self.btnSliceDisplay:Command.DisplaySlice,
            self.btnCutPlot:Command.PlotCut,
            self.btnCutPlotOver:Command.PlotOverCut,
            self.btnCutSaveToWorkspace:Command.SaveCutToWorkspace,
            self.btnCutSaveAscii:Command.SaveCutToAscii,
        }
        for button in self.button_commands.keys():
            button.clicked.connect(self.on_button_click)

    def on_button_click(self):
        sender = self.sender()
        self.presenter.notify(self.button_commands[sender])

    def get_workspace_to_load_path(self):
        return str(QtGui.QFileDialog.getOpenFileName(self,'Browse to Workspace File'))

    def display_loaded_workspaces(self, workspaces):
        #TODO preserve order??
        print '\nThe loaded workaspaces are'
        for workspace in workspaces:
            print workspace

if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    ui = MSliceGui()
    ui.show()
    app.exec_()
