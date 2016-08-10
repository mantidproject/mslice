from plot_window_ui import Ui_MainWindow
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatplotlibCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        self.axes.hold(False)
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PlotFigure(QtGui.QMainWindow,Ui_MainWindow):
    #TODO make a base class of common functionality
    def __init__(self,number,manager):
        self.number = number
        self._recieved_initial_focus = False
        super(PlotFigure,self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Figure %i'%number)
        self.canvas = MatplotlibCanvas(self)
        self.setCentralWidget(self.canvas)
        self._manager = manager

        self.menuKeep.aboutToShow.connect(self._report_as_kept_to_manager)
        self.menuMakeCurrent.aboutToShow.connect(self._report_as_current_to_manager)
        self.actionDump_To_Console.triggered.connect(self._dump_script_to_console)

        self._script_log = []
        self._import_aliases = {'plotting.pyplot': 'plt'} # the aliases used in script generation

        self.show() #this isn,t a good idea in non interactive mode #TODO FIX IT

    def closeEvent(self,event):
        self._manager.figure_closed(self.number)
        event.accept()

    def set_as_active(self):
        self.canvas.axes.hold(False)
        self._menubar_set("active")

    def set_as_kept(self):
        self.canvas.axes.hold(False)
        self._menubar_set("kept")

    def set_as_current(self):
        self.canvas.axes.hold(True)
        self._menubar_set("current")

    def _menubar_set(self,status):
        if status == "kept":
            self.menuKeep.setEnabled(False)
            self.menuMakeCurrent.setEnabled(True)
        elif status == "current":
            self.menuMakeCurrent.setEnabled(False)
            self.menuKeep.setEnabled(True)
        elif status == "active":
            self.menuKeep.setEnabled(True)
            self.menuMakeCurrent.setEnabled(True)
        else:
            raise ValueError("Invalid status %s"%status)

    def _report_as_kept_to_manager(self):
        self._manager.set_figure_as_kept(self.number)

    def _report_as_current_to_manager(self):
        self._manager.set_figure_as_current(self.number)

    def get_figure(self):
        return self.canvas.figure

    def script_log(self, source_module, function_name, call_args, call_kwargs):
        self._script_log.append((source_module, function_name, call_args, call_kwargs))
        print self._format_command(self._script_log[-1])

    def get_script(self):
        script = ""
        for library, alias in self._import_aliases.items():
            script += "import " + library + " as " + alias + "\n"
        for log in self._script_log:
            script += self._format_command(log) + "\n"
        return script

    def _format_command(self, command):
        """Return a line of python code for a tuple in the log"""
        output = ""
        source_module, function_name, call_args, call_kwargs = command
        if source_module in self._import_aliases.keys():
            source_module = self._import_aliases[source_module]

        if source_module:
            output += source_module + "."

        output += function_name + '('

        formatted_call_args = ", ".join(call_args)
        output += formatted_call_args

        call_kwargs = map(lambda x:"=".join(x), call_kwargs.items())
        formatted_call_kwargs = ", ".join(call_kwargs)

        if formatted_call_kwargs:
            if formatted_call_args:
                output += ", "
            output += formatted_call_kwargs
        output += ")"

        return output

    def _dump_script_to_console(self):
        print self.get_script()
