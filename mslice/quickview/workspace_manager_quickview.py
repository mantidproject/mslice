from __future__ import (absolute_import, division, print_function)
from quickview.QuickView import QuickView
from WorkspaceManagerPresenter import WorkspaceManagerPresenter
from workspacemanager.WorkspaceView import WorkspaceView
from MantidWorkspaceProvider import MantidWorkspaceProvider


class WorkspaceManagerQuickView(QuickView,WorkspaceView):
    def __init__(self,commands):
        super(WorkspaceManagerQuickView,self).__init__(commands)
        workspaceProvider = MantidWorkspaceProvider()
        self._presenter = WorkspaceManagerPresenter(self, workspaceProvider)

    def __getattribute__(self, item):
        # This is needed to handle calls to GUI functions generated on the fly correctly
        object.__getattr__(self, item)


if __name__ == '__main__':
    from command import Command as c
    m = WorkspaceManagerQuickView(c)
    m.app.exec_()
