from __future__ import (absolute_import, division, print_function)


from mslice.util.qt.QtWidgets import QApplication, QMainWindow, QLabel

from mslice.presenters.main_presenter import MainPresenter
from mslice.util.qt import load_ui
from mslice.views.mainview import MainView
from mslice.widgets.ipythonconsole.ipython_widget import IPythonWidget

TAB_2D = 0
TAB_EVENT = 1
TAB_HISTO = 2
TAB_SLICE = 1
TAB_CUT = 2
TAB_POWDER = 0

# ==============================================================================
# Classes
# ==============================================================================

class MainWindow(MainView, QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        load_ui(__file__, 'mainwindow.ui', self)
        self.init_ui()
        ipython = IPythonWidget()
        self.centralWidget().layout().addWidget(ipython)
        ipython.setFixedHeight(200)
        self.tabs = [self.wgtSlice, self.wgtCut, self.wgtPowder]
        self.tabs_to_show = {TAB_2D: [TAB_POWDER],
                             TAB_EVENT: [TAB_SLICE, TAB_CUT],
                             TAB_HISTO: []}
        workspace_presenter = self.wgtWorkspacemanager.get_presenter()
        dataloader_presenter = self.data_loading.get_presenter()
        slice_presenter = self.wgtSlice.get_presenter()
        powder_presenter = self.wgtPowder.get_presenter()
        cut_presenter = self.wgtCut.get_presenter()
        self._presenter = MainPresenter(self, workspace_presenter, dataloader_presenter,
                                        slice_presenter, powder_presenter, cut_presenter)

        workspace_provider = workspace_presenter.get_workspace_provider()
        dataloader_presenter.set_workspace_provider(workspace_provider)
        powder_presenter.set_workspace_provider(workspace_provider)
        slice_presenter.set_workspace_provider(workspace_provider)
        cut_presenter.set_workspace_provider(workspace_provider)

        self.wgtWorkspacemanager.tab_changed.connect(self.ws_tab_changed)

        self.wgtCut.error_occurred.connect(self.show_error)
        self.wgtSlice.error_occurred.connect(self.show_error)
        self.wgtWorkspacemanager.error_occurred.connect(self.show_error)
        self.wgtPowder.error_occurred.connect(self.show_error)
        self.data_loading.error_occurred.connect(self.show_error)
        self.wgtCut.busy.connect(self.show_busy)
        self.wgtSlice.busy.connect(self.show_busy)
        self.wgtWorkspacemanager.busy.connect(self.show_busy)
        self.wgtPowder.busy.connect(self.show_busy)

    def ws_tab_changed(self, tab):
        self.tabWidget_2.show()
        tab_to_show = self.tabs_to_show[tab]
        for tab_index in range(3):
            self.tabWidget_2.setTabEnabled(tab_index, False)
        if tab_to_show:
            self.tabWidget_2.setCurrentIndex(tab_to_show[0])
            for tab_index in tab_to_show:
                    self.tabWidget_2.setTabEnabled(tab_index, True)
        else:
            self.tabWidget_2.hide()

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
