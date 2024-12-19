from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QMenu, QStackedLayout

from mslice.models.units import EnergyUnits
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.main_presenter import MainPresenter
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.util.qt import load_ui
from mslice.views.interfaces.mainview import MainView
from mslice.widgets.ipythonconsole.ipython_widget import IPythonWidget
from mslice.widgets.workspacemanager import TAB_2D, TAB_EVENT, TAB_HISTO, TAB_NONPSD
from mslice.widgets.workspacemanager.command import Command as ws_command
from mslice.widgets.cut.command import Command as cut_command
from mslice.plotting.plot_window.plot_window import add_action

from functools import partial

TAB_SLICE = 1
TAB_CUT = 2
TAB_POWDER = 0

ERROR_STATUS_STYLESHEET = "color: red;"
WARNING_STATUS_STYLESHEET = "color: orange;"

# ==============================================================================
# Classes
# ==============================================================================


class MainWindow(MainView, QMainWindow):

    def __init__(self, in_mantid=False):
        QMainWindow.__init__(self)
        load_ui(__file__, 'mainwindow.ui', self)
        self.init_ui()

        self.tabs = [self.wgtSlice, self.wgtCut, self.wgtPowder]
        self.tabs_to_show = {TAB_2D: [TAB_POWDER],
                             TAB_EVENT: [TAB_SLICE, TAB_CUT],
                             TAB_HISTO: [],
                             TAB_NONPSD: [TAB_SLICE, TAB_CUT]}

        self.buttons_to_enable = {TAB_2D: [self.btnAdd, self.btnSubtract, self.composeFrame],
                                  TAB_EVENT: [self.btnMerge],
                                  TAB_HISTO: [self.btnPlot, self.btnOverplot],
                                  TAB_NONPSD: [self.btnAdd, self.btnSubtract, self.composeFrame]}
        if in_mantid:
            self.buttons_to_enable[TAB_HISTO] += [self.btnSaveToADS]
            self.btnSaveToADS.setText('Save to Workbench')

        self.stack_to_show = {TAB_2D: 1,
                              TAB_EVENT: 0,
                              TAB_HISTO: 0,
                              TAB_NONPSD: 1}

        self.composeCommand = {'Compose': ws_command.ComposeWorkspace,
                               'Scale': ws_command.Scale,
                               'Bose': ws_command.Bose}

        self.workspace_presenter = self.wgtWorkspacemanager.get_presenter()
        self.dataloader_presenter = self.data_loading.get_presenter()
        self.slice_plotter_presenter = SlicePlotterPresenter()
        slice_widget_presenter = self.wgtSlice.get_presenter()
        slice_widget_presenter.set_slice_plotter_presenter(self.slice_plotter_presenter)
        self.powder_presenter = self.wgtPowder.get_presenter()
        self.cut_plotter_presenter = CutPlotterPresenter()
        self.cut_widget_presenter = self.wgtCut.get_presenter()
        self.cut_widget_presenter.set_cut_plotter_presenter(self.cut_plotter_presenter)
        self.plot_selector_presenter = self.plot_selector_view.get_presenter()
        self.plot_selector_presenter.update_plot_list()
        self._presenter = MainPresenter(self, self.workspace_presenter, self.dataloader_presenter,
                                        slice_widget_presenter, self.powder_presenter, self.cut_widget_presenter,
                                        self.slice_plotter_presenter, self.cut_plotter_presenter,
                                        self.plot_selector_presenter)

        self.wgtWorkspacemanager.tab_changed.connect(self.ws_tab_changed)
        self.setup_save()
        self.setup_compose_button()
        self.btnSave.clicked.connect(self.button_save)
        self.btnAdd.clicked.connect(self.button_add)
        self.btnRename.clicked.connect(self.button_rename)
        self.btnSubtract.clicked.connect(self.button_subtract)
        self.btnDelete.clicked.connect(self.button_delete)
        self.btnMerge.clicked.connect(self.button_merge)
        self.btnPlot.clicked.connect(self.button_plot)
        self.btnOverplot.clicked.connect(self.button_overplot)
        self.btnSaveToADS.clicked.connect(self.button_savetoads)
        self.btnCompose.clicked.connect(self.button_compose)
        self.ws_tab_changed(0)

        self.wgtCut.warning_occurred.connect(self.show_warning)

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
        self.action_quit.triggered.connect(self.close)

        self._en_default_actions = []
        for e_unit in EnergyUnits.get_all_units():
            action = add_action(self.menuDefault_Energy_Units, self, e_unit, checkable=True, visible=True)
            action.triggered.connect(partial(self.set_energy_default, action))
            self._en_default_actions.append(action)
        self._en_default_actions[0].setChecked(True)

        self.actionEUnitConvEnabled.triggered.connect(partial(self.set_energy_conversion, True))
        self.actionEUnitConvDisabled.triggered.connect(partial(self.set_energy_conversion, False))
        self._cut_algo_map = {'Rebin': self.actionCutAlgoRebin, 'Integration': self.actionCutAlgoIntegration}
        self.actionCutAlgoRebin.triggered.connect(partial(self.set_cut_algorithm_default, 'Rebin'))
        self.actionCutAlgoIntegration.triggered.connect(partial(self.set_cut_algorithm_default, 'Integration'))

        self.print_startup_notifications()

    def setup_save(self):
        menu = QMenu(self.btnSave)
        menu.addAction("Nexus (*.nxs)", lambda: self.button_save('Nexus'))
        menu.addAction("NXSPE (*.nxspe)", lambda: self.button_save('NXSPE'))
        menu.addAction("ASCII (*.txt)", lambda: self.button_save('Ascii'))
        menu.addAction("Matlab (*.mat)", lambda: self.button_save('Matlab'))
        self.btnSave.setMenu(menu)

    def setup_compose_button(self):
        self.stackLayout = QStackedLayout(self.stackFrame)
        self.stackLayout.addWidget(self.btnSaveToADS)
        self.stackLayout.addWidget(self.composeFrame)
        self.stackFrame.setLayout(self.stackLayout)
        menu = QMenu(self.btnComposeMenu)
        menu.addAction("Scale", lambda: self.button_compose('Scale'))
        menu.addAction("Bose", lambda: self.button_compose('Bose'))
        self.btnComposeMenu.setMenu(menu)
        self.btnComposeMenu.setMaximumWidth(10)

    def change_main_tab(self, tab):
        self.tabWidget.setCurrentIndex(tab)

    def ws_tab_changed(self, tab):
        self.enable_widget_tabs(tab)
        self.enable_buttons(tab)

    def enable_widget_tabs(self, workspace_tab):
        """Enables correct powder/slice/cut tabs based on workspace tab"""
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
        """Enables correct buttons based on workspace tab"""
        self.stackLayout.setCurrentIndex(self.stack_to_show[tab])
        variable_buttons = [self.btnAdd, self.btnSubtract, self.btnMerge, self.btnPlot, self.btnOverplot,
                            self.btnSaveToADS, self.composeFrame]
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
        self.cut_widget_presenter.notify(cut_command.PlotFromWorkspace)

    def button_overplot(self):
        self.cut_widget_presenter.notify(cut_command.PlotOverFromWorkspace)

    def button_savetoads(self):
        self.workspace_presenter.notify(ws_command.SaveToADS)

    def button_compose(self, value=False):
        if value:
            self.btnCompose.setText(value)
        self.workspace_presenter.notify(self.composeCommand[str(self.btnCompose.text())])

    def init_ui(self):
        self.setup_ipython()
        self.busy_text = QLabel()
        self.statusBar().addPermanentWidget(self.busy_text)
        self.busy_text.setText("  Idle  ")
        self.busy_text.setStyleSheet("QLabel { color: black }")
        self.busy = False

    def setup_ipython(self):
        ipython = IPythonWidget()
        self._console = ipython
        self.splitter.addWidget(ipython)
        self.splitter.setSizes([500, 250])

    def show_warning(self, msg):
        """Show a warning message on status bar. If msg ==""  the function will clear the displayed message """
        self.statusbar.setStyleSheet(WARNING_STATUS_STYLESHEET)
        self.statusbar.showMessage(msg)

    def show_error(self, msg):
        """Show an error message on status bar. If msg ==""  the function will clear the displayed message """
        self.statusbar.setStyleSheet(ERROR_STATUS_STYLESHEET)
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

    def get_energy_default(self):
        return [action.text() for action in self._en_default_actions if action.isChecked()][0]

    def set_energy_default(self, this_action):
        if this_action.isChecked():
            for action in self._en_default_actions:
                # Only one unit can be set as default at a time
                if action == this_action:
                    action.setChecked(True)
                else:
                    action.setChecked(False)
        else:
            # User unchecked this action - either go back to first or set another action as default
            if this_action == self._en_default_actions[0]:
                self._en_default_actions[1].setChecked(True)
            else:
                self._en_default_actions[0].setChecked(True)
        self._presenter.set_energy_default(self.get_energy_default())

    def set_energy_conversion(self, EnabledClicked):
        if EnabledClicked:
            self.actionEUnitConvDisabled.setChecked(not self.actionEUnitConvEnabled.isChecked())
        else:
            self.actionEUnitConvEnabled.setChecked(not self.actionEUnitConvDisabled.isChecked())

    def set_cut_algorithm_default(self, algo):
        for action in self._cut_algo_map.keys():
            if algo not in action:
                self._cut_algo_map[action].setChecked(False)
        self._presenter.set_cut_algorithm_default(algo)

    def is_energy_conversion_allowed(self):
        return self.actionEUnitConvEnabled.isChecked()

    def print_startup_notifications(self):
        # if notifications are required to be printed on mslice start up, add to list.

        # Disable this notification pending decision on changing the default
        # print_list = ["\nWARNING: The default cut algorithm in mslice has been changed " \
        #               "from 'Rebin (average counts)' to 'Intergration (summed counts)'.\n" \
        #               "This is expected to result in different output values to those obtained historically.\n" \
        #               "For more information, please refer to documentation at: " \
        #               "https://mantidproject.github.io/mslice/cutting.html"]
        print_list = []

        for item in print_list:
            for strn in item.split('\n'):
                self._console.execute(f'print("{strn}")', hidden=True)

    def closeEvent(self, event):
        self._console.cleanup()
