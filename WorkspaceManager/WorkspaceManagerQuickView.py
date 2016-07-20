from QuickView.QuickView import QuickView
from command import Command
from WorkspaceManagerPresenter import WorkspaceManagerPresenter
from MantidWorkspaceProvider import MantidWorkspaceProvider

class WorkspaceManagerQuickView(QuickView):
    def __init__(self,_commands):
        super(WorkspaceManagerQuickView,self).__init__(_commands)
        workspaceProvider = MantidWorkspaceProvider()
        self._presenter = WorkspaceManagerPresenter(self, workspaceProvider)


if __name__ == '__main__':

    m = WorkspaceManagerQuickView(Command)
    m.app.exec_()


