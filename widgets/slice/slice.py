from slice_ui import Ui_Form
from PyQt4.QtGui import QWidget, QMessageBox
from slice_plotter_view import SlicePlotterView
from slice_plotter_presenter import SlicePlotterPresenter
from matplotlib_slice_plotter import MatplotlibSlicePlotter
from command import Command


class SliceWidget(QWidget, Ui_Form, SlicePlotterView):
    def __init__(self, main_window, *args, **kwargs):
        """This Widget provides basic control over displaying slices. This widget is NOT USABLE without a main window

        The main window must implement MainView"""
        super(SliceWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnSliceDisplay.clicked.connect(self._btn_clicked)
        self.display_errors_to_statusbar = True

    def _btn_clicked(self):
        self._presenter.notify(Command.DisplaySlice)

    def set_main_window(self,main_window):
        self._presenter = SlicePlotterPresenter(main_view=main_window, slice_view=self,
                                                slice_plotter=MatplotlibSlicePlotter())
        self._main_window = main_window

    def _display_error(self, error_string, timeout_ms):
        if self.display_errors_to_statusbar:
            self._main_window.statusBar().showMessage(error_string, timeout_ms)
        else:
            m = QMessageBox()
            m.setWindowTitle('MSlice Error Message')
            m.setText(error_string)
            m.exec_()

    def error_select_one_workspace(self):
        self._display_error('Please select a workspace to slice', 2000)
