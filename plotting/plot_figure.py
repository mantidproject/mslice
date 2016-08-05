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
    def __init__(self,number,manager):
        self.number = number
        self._recieved_initial_focus = False
        super(PlotFigure,self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Figure %i'%number)
        self.canvas = MatplotlibCanvas(self)
        self.setCentralWidget(self.canvas)
        self._manager = manager # TODO askMartin. avoiding circular imports.is this OK

        self.menuKeep.aboutToShow.connect(self._report_as_kept_to_manager)
        self.menuMakeCurrent.aboutToShow.connect(self._report_as_current_to_manager)

        self.show() #this isnt a good idea in non interactive mode

    def closeEvent(self,event):
        self._manager.figure_closed(self.number)
        event.accept()

    def windowActivationChange(self, *args, **kwargs):
        #this event may happen before my PlotFigure.__init__ has finished, be careful
        QtGui.QMainWindow.windowActivationChange(self, *args, **kwargs)
        return
        # This Feature is not ready for demo and is too risky
        #We have to skip the first one because it happens before we have properly got things set up
        if not self._recieved_initial_focus:
            self._recieved_initial_focus = True
            return
        if self.isActiveWindow():
            self._manager.set_figure_as_active(self.number)  # window has just recieved focus. make it the active plot

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

    def gca(self):
        return self.canvas.figure.gca()

    def sca(self, *args, **kwargs):
        return self.canvas.figure.sca(*args,**kwargs)

    def _gci(self):
        return self.gca()._gci()

    def colorbar(self,*args, **kwargs):
        return self.canvas.figure.colorbar(*args,**kwargs)
