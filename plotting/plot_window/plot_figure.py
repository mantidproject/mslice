from plotting.plot_window.plot_window_ui import Ui_MainWindow
from base_qt_plot_window import BaseQtPlotWindow
from plotting.plot_window.plot_window_ui import Ui_MainWindow


class PlotFigure(BaseQtPlotWindow, Ui_MainWindow):
    def __init__(self,number,manager):
        super(PlotFigure,self).__init__(number, manager)
        self.menuKeep.aboutToShow.connect(self._report_as_kept_to_manager)
        self.menuMakeCurrent.aboutToShow.connect(self._report_as_current_to_manager)
        self.actionDump_To_Console.triggered.connect(self._dump_script_to_console)

        self.actionZoom_In.triggered.connect(self.canvas.zoom_in)
        self.actionZoom_Out.triggered.connect(self.canvas.zoom_out)

    def _display_status(self,status):
        if status == "kept":
            self.menuKeep.setEnabled(False)
            self.menuMakeCurrent.setEnabled(True)
        elif status == "current":
            self.menuMakeCurrent.setEnabled(False)
            self.menuKeep.setEnabled(True)
