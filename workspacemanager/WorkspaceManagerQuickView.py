from quickview.QuickView import QuickView
from WorkspaceManagerPresenter import WorkspaceManagerPresenter
from MantidWorkspaceProvider import MantidWorkspaceProvider
import time


class WorkspaceManagerQuickView(QuickView):
    def __init__(self,commands):
        super(WorkspaceManagerQuickView,self).__init__(commands)
        workspaceProvider = MantidWorkspaceProvider()
        self._presenter = WorkspaceManagerPresenter(self, workspaceProvider)


if __name__ == '__main__':
    from command import Command as c
    m = WorkspaceManagerQuickView(c)
    m.app.exec_()


