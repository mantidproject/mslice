from quickview.QuickView import QuickView
from WorkspaceManagerPresenter import WorkspaceManagerPresenter
from workspacemanager.WorkspaceView import WorkspaceView
from MantidWorkspaceProvider import MantidWorkspaceProvider
import time


class WorkspaceManagerQuickView(QuickView,WorkspaceView):
    def __init__(self,commands):
        super(WorkspaceManagerQuickView,self).__init__(commands)
        workspaceProvider = MantidWorkspaceProvider()
        self._presenter = WorkspaceManagerPresenter(self, workspaceProvider)
        print 'wsmquick created'

    def __getattribute__(self, item):
       # print 'wsmqv gettatr ', item
        if item in ("get_presenter",):
            print 'WSM PRESENTER'+WorkspaceView.__getattr__(self,item)
            return WorkspaceView.__getattr__(self,item)
        else:
            object.__getattr__(self, item)


if __name__ == '__main__':
    from command import Command as c
    m = WorkspaceManagerQuickView(c)
    m.app.exec_()


