from __future__ import (absolute_import, division, print_function)

from mslice.util.qt.QtWidgets import QApplication, QMainWindow, QLabel, QMenu

from mslice.presenters.main_presenter import MainPresenter
from mslice.util.qt import load_ui
from mslice.views.mainview import MainView
from mslice.widgets.workspacemanager import TAB_2D, TAB_EVENT, TAB_HISTO, TAB_NONPSD
from mslice.widgets.workspacemanager.command import Command as ws_command
from mslice.widgets.cut.command import Command as cut_command

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

        self.tabs = [self.wgtSlice, self.wgtCut, self.wgtPowder]
        self.tabs_to_show = {TAB_2D: [TAB_POWDER],
                             TAB_EVENT: [TAB_SLICE, TAB_CUT],
                             TAB_HISTO: [],
                             TAB_NONPSD: [TAB_SLICE, TAB_CUT]}

        self.buttons_to_enable = {TAB_2D: [self.btnAdd, self.btnSubtract],
                                  TAB_EVENT: [self.btnMerge],
                                  TAB_HISTO: [self.btnPlot, self.btnOverplot],
                                  TAB_NONPSD: [self.btnAdd, self.btnSubtract]}

        self.workspace_presenter = self.wgtWorkspacemanager.get_presenter()
        dataloader_presenter = self.data_loading.get_presenter()
        slice_presenter = self.wgtSlice.get_presenter()
        powder_presenter = self.wgtPowder.get_presenter()
        self.cut_presenter = self.wgtCut.get_presenter()
        self._presenter = MainPresenter(self, self.workspace_presenter, dataloader_presenter,
                                        slice_presenter, powder_presenter, self.cut_presenter)

        self.wgtWorkspacemanager.tab_changed.connect(self.ws_tab_changed)
        self.setup_save()
        self.btnSave.clicked.connect(self.button_save)
        self.btnAdd.clicked.connect(self.button_add)
        self.btnRename.clicked.connect(self.button_rename)
        self.btnSubtract.clicked.connect(self.button_subtract)
        self.btnDelete.clicked.connect(self.button_delete)
        self.btnMerge.clicked.connect(self.button_merge)
        self.btnPlot.clicked.connect(self.button_plot)
        self.btnOverplot.clicked.connect(self.button_overplot)
        self.btnHistory.hide()
        self.ws_tab_changed(0)

        self.wgtCut.error_occurred.connect(self.show_error)
        self.wgtSlice.error_occurred.connect(self.show_error)
        self.wgtWorkspacemanager.error_occurred.connect(self.show_error)
        self.wgtPowder.error_occurred.connect(self.show_error)
        self.data_loading.error_occurred.connect(self.show_error)
        self.wgtCut.busy.connect(self.show_busy)
        self.wgtSlice.busy.connect(self.show_busy)
        self.wgtWorkspacemanager.busy.connect(self.show_busy)
        self.wgtPowder.busy.connect(self.show_busy)
        self.data_loading.busy.connect(self.show_busy)
        self.actionQuit.triggered.connect(self.close)

    def setup_save(self):
        menu = QMenu()
        menu.addAction("Nexus (*.nxs)", lambda: self.button_save('Nexus'))
        menu.addAction("ASCII (*.txt)", lambda: self.button_save('Ascii'))
        menu.addAction("Matlab (*.mat)", lambda: self.button_save('Matlab'))
        self.btnSave.setMenu(menu)

    def change_main_tab(self, tab):
        self.tabWidget.setCurrentIndex(tab)

    def ws_tab_changed(self, tab):
        self.enable_widget_tabs(tab)
        self.enable_buttons(tab)

    def enable_widget_tabs(self, workspace_tab):
        '''enables correct powder/slice/cut tabs based on workspace tab'''
        self.tabWidget_2.show()
        tab_to_show = self.tabs_to_show[workspace_tab]
        for tab_index in range(3):
            self.tabWidget_2.setTabEnabled(tab_index, False)
        if tab_to_show:
            self.tabWidget_2.setCurrentIndex(tab_to_show[0])
            for tab_index in tab_to_show:
                    self.tabWidget_2.setTabEnabled(tab_index, True)
        else:
            self.tabWidget_2.hide()

    def enable_buttons(self, tab):
        '''enables correct buttons based on workspace tab'''
        variable_buttons = [self.btnAdd, self.btnSubtract, self.btnMerge, self.btnPlot, self.btnOverplot]
        for button in variable_buttons:
            button.hide()
        for button in self.buttons_to_enable[tab]:
            button.show()

    def button_subtract(self):
        self.workspace_presenter.notify(ws_command.Subtract)

    def button_save(self, file_type):
        self.workspace_presenter.notify(getattr(ws_command, 'SaveSelectedWorkspace' + file_type))

    def button_add(self):
        self.workspace_presenter.notify(ws_command.Add)

    def button_rename(self):
        self.workspace_presenter.notify(ws_command.RenameWorkspace)

    def button_delete(self):
        self.workspace_presenter.notify(ws_command.RemoveSelectedWorkspaces)

    def button_merge(self):
        self.workspace_presenter.notify(ws_command.CombineWorkspace)

    def button_plot(self):
        self.cut_presenter.notify(cut_command.PlotFromWorkspace)

    def button_overplot(self):
        self.cut_presenter.notify(cut_command.PlotOverFromWorkspace)

    def init_ui(self):
        self.busy_text = QLabel()
        self.statusBar().addPermanentWidget(self.busy_text)
        self.busy_text.setText("  Idle  ")
        self.busy_text.setStyleSheet("QLabel { color: black }")
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
