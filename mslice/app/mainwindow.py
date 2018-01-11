from __future__ import (absolute_import, division, print_function)

from qtpy.QtWidgets import QApplication, QMainWindow, QLabel

from mslice.presenters.main_presenter import MainPresenter
from mslice.util.qt import load_ui
from mslice.views.mainview import MainView


# ==============================================================================
# Classes
# ==============================================================================

class MainWindow(MainView, QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        load_ui(__file__, 'mainwindow.ui', self)
        self.init_ui()

        workspace_presenter = self.wgtWorkspacemanager.get_presenter()
        slice_presenter = self.wgtSlice.get_presenter()
        powder_presenter = self.wgtPowder.get_presenter()
        cut_presenter = self.wgtCut.get_presenter()
        self._presenter = MainPresenter(self, workspace_presenter, slice_presenter , powder_presenter, cut_presenter)

        workspace_provider = workspace_presenter.get_workspace_provider()
        powder_presenter.set_workspace_provider(workspace_provider)
        slice_presenter.set_workspace_provider(workspace_provider)
        cut_presenter.set_workspace_provider(workspace_provider)

        self.wgtCut.error_occurred.connect(self.show_error)
        self.wgtSlice.error_occurred.connect(self.show_error)
        self.wgtWorkspacemanager.error_occurred.connect(self.show_error)
        self.wgtPowder.error_occurred.connect(self.show_error)
        self.wgtCut.busy.connect(self.show_busy)
        self.wgtSlice.busy.connect(self.show_busy)
        self.wgtWorkspacemanager.busy.connect(self.show_busy)
        self.wgtPowder.busy.connect(self.show_busy)

    def init_ui(self):
        self.busy_text = QLabel()
        self.statusBar().addPermanentWidget(self.busy_text)
        self.busy_text.setText("  Idle  ")
        self.busy = False

    def show_error(self, msg):
        """Show an error message on status bar. If msg ==""  the function will clear the displayed message """
        self.statusbar.showMessage(msg)

    def get_presenter(self):
        return self._presenter

    def show_busy(self, busy):
        if busy and not self.busy:
            self.busy = True
            self.busy_text.setStyleSheet("QLabel { color: red }")
            self.busy_text.setText("  Busy  ")
        elif not busy and self.busy:
            self.busy = False
            self.busy_text.setStyleSheet("QLabel { color: black }")
            self.busy_text.setText("  Idle  ")
        else:
            return
        QApplication.processEvents()
