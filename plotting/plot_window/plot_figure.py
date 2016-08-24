from base_qt_plot_window import BaseQtPlotWindow
from plotting.plot_window.plot_window_ui import Ui_MainWindow
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT

class PlotFigure(BaseQtPlotWindow, Ui_MainWindow):
    def __init__(self,number,manager):
        super(PlotFigure,self).__init__(number, manager)
        self.menuKeep.aboutToShow.connect(self._report_as_kept_to_manager)
        self.menuMakeCurrent.aboutToShow.connect(self._report_as_current_to_manager)
        self.actionDump_To_Console.triggered.connect(self._dump_script_to_console)
        self.stock_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.stock_toolbar.hide()
        self.show() #is not a good idea in non interactive mode


    def _display_status(self,status):
        if status == "kept":
            self.menuKeep.setEnabled(False)
            self.menuMakeCurrent.setEnabled(True)
        elif status == "current":
            self.menuMakeCurrent.setEnabled(False)
            self.menuKeep.setEnabled(True)
