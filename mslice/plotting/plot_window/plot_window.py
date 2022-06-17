from __future__ import (absolute_import, division, print_function)

from qtpy import QtCore, QtWidgets

from matplotlib.figure import Figure
from mantidqt.icons import get_icon

from mslice.plotting.backend import get_canvas_and_toolbar_cls
from mslice.util.qt.validator_helper import double_validator_without_separator

FigureCanvas, NavigationToolbar2QT = get_canvas_and_toolbar_cls()


class MatplotlibQTCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, manager, width=5, height=4, dpi=100, parent=None):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.manager = manager

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PlotWindow(QtWidgets.QMainWindow):
    """The plot window that holds a matplotlib figure"""

    redraw = QtCore.Signal()

    def __init__(self, manager, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.setup_ui(manager)
        self._first_time_show = False

    def closeEvent(self, _):
        self.lose_waterfall_x_edt_focus()  # lose focus so toggle_waterfall does not trigger upon close
        self.lose_waterfall_y_edt_focus()
        self.canvas.manager.window_closing()

    def showEvent(self, evt):
        if not self._first_time_show:
            try:
                self.canvas.manager.plot_handler.save_default_options()
            except AttributeError:
                pass
            self._first_time_show = True
        super(PlotWindow, self).showEvent(evt)

    def setup_ui(self, manager):
        # canvas
        self.canvas = MatplotlibQTCanvas(manager, parent=self)
        self.setCentralWidget(self.canvas)
        # stock toolbar is used behind the scenes to handle zooming
        self.stock_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.stock_toolbar.hide()

        self.create_menus()
        self.create_toolbar()
        self.create_status_bar()

    def create_menus(self):
        self.menubar = self.menuBar()

        # File
        self.menu_file = QtWidgets.QMenu('File', self.menubar)
        self.menubar.addMenu(self.menu_file)
        self.add_file_actions(self.menu_file)

        # information
        self.menu_information = QtWidgets.QMenu("Information", self.menubar)
        self.menubar.addMenu(self.menu_information)
        self.menu_recoil_lines = QtWidgets.QMenu("Recoil lines", self.menu_information)
        self.add_information_actions(self.menu_recoil_lines,
                                     ("Hydrogen", "Deuterium", "Helium", "Arbitrary Nuclei"))
        self.menu_information.addMenu(self.menu_recoil_lines)
        self.menu_bragg_peaks = QtWidgets.QMenu("Bragg peaks", self.menu_information)
        self.add_information_actions(self.menu_bragg_peaks,
                                     ("Aluminium", "Copper", "Niobium", "Tantalum", "CIF file"))
        self.menu_information.addMenu(self.menu_bragg_peaks)

        # intensity
        self.menu_intensity = QtWidgets.QMenu("Intensity", self.menubar)
        self.menubar.addMenu(self.menu_intensity)
        self.add_intensity_actions(self.menu_intensity)

        self.setMenuBar(self.menubar)
        self.menu_information.setDisabled(True)
        self.menu_intensity.setDisabled(True)

    def add_file_actions(self, menu):
        self.action_gen_script = add_action(menu, self, "Generate Script File")
        menu.addAction(self.action_gen_script)
        self.action_gen_script_clipboard = add_action(menu, self, "Generate Script to Clipboard")
        menu.addAction(self.action_gen_script_clipboard)

        self.action_quit = add_action(menu, self, "Close", visible=True)
        menu.addAction(self.action_quit)
        self.action_quit.triggered.connect(self.close)

    def add_information_actions(self, menu, items):
        for text in items:
            action = add_action(menu, self, text, checkable=True, visible=True)
            setattr(self, create_attribute_name(text), action)
            menu.addAction(action)

    def add_intensity_actions(self, menu):
        self.action_sqe = add_action(menu, self, "S(Q,E)", checkable=True, checked=True, visible=True)
        menu.addAction(self.action_sqe)
        self.action_chi_qe = add_action(menu, self, "Chi''(Q,E)", checkable=True, visible=True)
        menu.addAction(self.action_chi_qe)
        self.action_chi_qe_magnetic = add_action(menu, self, "Chi''(Q,E) magnetic", checkable=True, visible=True)
        menu.addAction(self.action_chi_qe_magnetic)
        self.action_d2sig_dw_de = add_action(menu, self, "d2sigma/dOmega.dE", checkable=True, visible=True)
        menu.addAction(self.action_d2sig_dw_de)
        self.action_symmetrised_sqe = add_action(menu, self, "Symmetrised S(Q,E)", checkable=True, visible=True)
        menu.addAction(self.action_symmetrised_sqe)
        self.action_gdos = add_action(menu, self, "GDOS", checkable=True, visible=True)
        menu.addAction(self.action_gdos)

    def create_toolbar(self):
        self.toolbar = QtWidgets.QToolBar()
        self.add_toolbar_actions(self.toolbar)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

    def add_toolbar_actions(self, toolbar):
        self.action_zoom_in = add_action(toolbar, self,  "Zoom In", on_triggered=self.stock_toolbar.zoom,
                                         icon_name='mdi.magnify-plus-outline', checkable=True, visible=True)
        self.action_zoom_out = add_action(toolbar, self,  "Zoom Out", on_triggered=self.stock_toolbar.back,
                                          icon_name='mdi.magnify-minus-outline', checkable=False, visible=True)
        self.action_toggle_legends = add_action(toolbar, self, "Legends", checkable=True,
                                                checked=True)
        toolbar.addSeparator()
        self.action_keep = add_action(toolbar, self,  "Keep", checkable=True, visible=True)
        self.action_make_current = add_action(toolbar, self,  "Make Current",
                                              checkable=True, checked=True, visible=True)
        self.keep_make_current_group = QtWidgets.QActionGroup(self)
        self.keep_make_current_group.addAction(self.action_keep)
        self.keep_make_current_group.addAction(self.action_make_current)
        self.keep_make_current_seperator = toolbar.addSeparator()

        self.action_save_image = add_action(toolbar, self, "Save Image", icon_name='mdi.content-save')
        self.action_copy_image = add_action(toolbar, self, "Copy Image", icon_name='mdi.content-copy', visible=True)
        self.action_print_plot = add_action(toolbar, self,  "Print", icon_name='mdi.printer', visible=True)
        self.action_plot_options = add_action(toolbar, self, "Plot Options", icon_name='mdi.settings')

        toolbar.addSeparator()
        self.action_waterfall = add_action(toolbar, self, "Waterfall", checkable=True)
        self.add_waterfall_edit(self.toolbar)
        self.action_waterfall.triggered.connect(self.toggle_waterfall_edit)
        self.action_waterfall.changed.connect(self.toggle_waterfall_edit)
        self.action_interactive_cuts = add_action(toolbar, self,  "Interactive Cuts", checkable=True)
        # options for interactive cuts only
        self.action_save_cut = add_action(toolbar, self,  "Save Cut to Workspace")
        self.action_flip_axis = add_action(toolbar, self,  "Flip Integration Axis",
                                           icon_name='mdi.twitter-retweet')

    def add_waterfall_edit(self, parent):
        self.waterfall_x_lbl = QtWidgets.QLabel("x:", parent)
        self.waterfall_y_lbl = QtWidgets.QLabel("y:", parent)
        self.waterfall_x_edt = QtWidgets.QLineEdit("0", parent)
        self.waterfall_y_edt = QtWidgets.QLineEdit("0", parent)
        sz = self.waterfall_x_edt.sizeHint()
        sz.setWidth(60)
        self.waterfall_x_edt.setMaximumSize(sz)
        self.waterfall_y_edt.setMaximumSize(sz)
        validator = double_validator_without_separator()
        self.waterfall_x_edt.setValidator(validator)
        self.waterfall_y_edt.setValidator(validator)
        self.waterfall_x_lbl_act = parent.addWidget(self.waterfall_x_lbl)
        self.waterfall_x_edt_act = parent.addWidget(self.waterfall_x_edt)
        self.waterfall_y_lbl_act = parent.addWidget(self.waterfall_y_lbl)
        self.waterfall_y_edt_act = parent.addWidget(self.waterfall_y_edt)
        self.waterfall_x_lbl_act.setVisible(False)
        self.waterfall_y_lbl_act.setVisible(False)
        self.waterfall_x_edt_act.setVisible(False)
        self.waterfall_y_edt_act.setVisible(False)

    def toggle_waterfall_edit(self):
        is_waterfall = self.waterfall
        self.waterfall_x_lbl_act.setVisible(is_waterfall)
        self.waterfall_y_lbl_act.setVisible(is_waterfall)
        self.waterfall_x_edt_act.setVisible(is_waterfall)
        self.waterfall_y_edt_act.setVisible(is_waterfall)

    def lose_waterfall_x_edt_focus(self):
        self._lose_focus_without_signal(self.waterfall_x_edt)

    def lose_waterfall_y_edt_focus(self):
        self._lose_focus_without_signal(self.waterfall_y_edt)

    @staticmethod
    def _lose_focus_without_signal(qt_obj):
        qt_obj.blockSignals(True)  # prevents a subsequent call to this function
        qt_obj.clearFocus()
        qt_obj.blockSignals(False)

    def create_status_bar(self):
        self.statusbar = QtWidgets.QStatusBar(self)
        self.stock_toolbar.message.connect(self.statusbar.showMessage)
        self.setStatusBar(self.statusbar)

    def flag_as_kept(self):
        # The QActionGroup ensures the other is not checked
        self.action_keep.setChecked(True)

    def flag_as_current(self):
        # The QActionGroup ensures the other is not checked
        self.action_make_current.setChecked(True)

    def display_error(self, error_text):
        # Displays an error message box
        err_box = QtWidgets.QErrorMessage(self)
        err_box.showMessage(error_text)

    def disable_action(self, key):
        """
        Disable the action based on the string key
        :param key: A string denoting the action. Spaces are
                    replaced by _ and the whole thing is converted to lowercase
        """
        try:
            getattr(self, create_attribute_name(key)).setChecked(False)
        except AttributeError:
            pass

    def uncheck_action_by_text(self, holder, item_text):
        for action in holder.actions():
            if action.iconText() == item_text and action.isCheckable() and action.isChecked():
                action.setChecked(False)

    @property
    def waterfall(self):
        return (self.action_waterfall.isChecked()
                and self.action_waterfall.isVisible()
                and self.action_waterfall.isEnabled())

    @waterfall.setter
    def waterfall(self, value):
        self.action_waterfall.setChecked(value)
        self.toggle_waterfall_edit()

    @property
    def waterfall_x(self):
        return float(self.waterfall_x_edt.text())

    @waterfall_x.setter
    def waterfall_x(self, value):
        self.waterfall_x_edt.setText(str(value))

    @property
    def waterfall_y(self):
        return float(self.waterfall_y_edt.text())

    @waterfall_y.setter
    def waterfall_y(self, value):
        self.waterfall_y_edt.setText(str(value))

def create_attribute_name(text):
    """Create the name of an action attribute based on the text"""
    return "action_" + text.replace(" ", "_").lower()


def add_action(holder, parent, text, on_triggered=None, icon_name=None,
               checkable=False, checked=False, visible=False):
    """Create a new action based on the given attributes and add it to the given
    holder"""
    action = QtWidgets.QAction(text, parent)
    if icon_name is not None:
        action.setIcon(get_icon(icon_name))
    action.setCheckable(checkable)
    action.setChecked(checked)
    action.setVisible(visible)
    if on_triggered is not None:
        action.triggered.connect(on_triggered)
    holder.addAction(action)
    return action
