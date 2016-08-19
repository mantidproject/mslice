from PyQt4.QtGui import QWidget, QMessageBox

from command import Command
from models.slice.matplotlib_slice_plotter import MatplotlibSlicePlotter
from presenters.slice_plotter_presenter import SlicePlotterPresenter
from slice_ui import Ui_Form
from views.slice_plotter_view import SlicePlotterView


class SliceWidget(QWidget, Ui_Form, SlicePlotterView):
    def __init__(self, main_window, *args, **kwargs):
        """This Widget provides basic control over displaying slices. This widget is NOT USABLE without a main window

        The main window must implement MainView"""
        super(SliceWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnSliceDisplay.clicked.connect(self._btn_clicked)
        self.display_errors_to_statusbar = True
        self._presenter = SlicePlotterPresenter(self, MatplotlibSlicePlotter())

    def _btn_clicked(self):
        self._presenter.notify(Command.DisplaySlice)

    def _display_error(self, error_string, timeout_ms):
        # Should be replaced to emit signal containing the error message
        if self.display_errors_to_statusbar:
            self._main_window.statusBar().showMessage(error_string, timeout_ms)
        else:
            m = QMessageBox()
            m.setWindowTitle('MSlice Error Message')
            m.setText(error_string)
            m.exec_()

    def error_select_one_workspace(self):
        self._display_error('Please select a workspace to slice', 2000)

    def get_presenter(self):
        return self._presenter
