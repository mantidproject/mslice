from __future__ import (absolute_import, division, print_function)

import importlib

from matplotlib.figure import Figure

from mslice.util.qt import QT_VERSION
from mslice.util.qt import QtCore, QtWidgets
import qtawesome as qta

# The FigureCanvas & Toolbar are QWidgets so we must import it from the mpl backend that matches
# the version of Qt we are running with
mpl_qt_backend  = importlib.import_module('matplotlib.backends.backend_qt{}agg'.format(QT_VERSION[0]))
FigureCanvasQTAgg = getattr(mpl_qt_backend, 'FigureCanvasQTAgg')
NavigationToolbar2QT = getattr(mpl_qt_backend, 'NavigationToolbar2QT')


class MatplotlibQTCanvas(FigureCanvasQTAgg):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, manager, width=5, height=4, dpi=100, parent=None):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvasQTAgg.__init__(self, self.figure)
        self.setParent(parent)
        self.manager = manager

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)


class PlotWindow(QtWidgets.QMainWindow):
    """The plot window that holds a matplotlib figure"""

    def __init__(self, manager, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.setup_ui(manager)

    def closeEvent(self, _):
        self.canvas.manager.window_closing()

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

    def add_information_actions(self, menu, items):
        for text in items:
            action = add_action(menu, self, text, checkable=True)
            setattr(self, create_attribute_name(text), action)
            menu.addAction(action)

    def add_intensity_actions(self, menu):
        self.action_sqe = add_action(menu, self, "S(Q,E)", checkable=True, checked=True)
        menu.addAction(self.action_sqe)
        self.action_chi_qe = add_action(menu, self, "Chi''(Q,E)", checkable=True)
        menu.addAction(self.action_chi_qe)
        self.action_chi_qe_magnetic = add_action(menu, self, "Chi''(Q,E) magnetic", checkable=True)
        menu.addAction(self.action_chi_qe_magnetic)
        self.action_d2sig_dw_de = add_action(menu, self, "d2sigma/dOmega.dE", checkable=True)
        menu.addAction(self.action_d2sig_dw_de)
        self.action_symmetrised_sqe = add_action(menu, self, "Symmetrised S(Q,E)", checkable=True)
        menu.addAction(self.action_symmetrised_sqe)
        self.action_gdos = add_action(menu, self, "GDOS", checkable=True)
        menu.addAction(self.action_gdos)

    def create_toolbar(self):
        self.toolbar = QtWidgets.QToolBar()
        self.add_toolbar_actions(self.toolbar)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)

    def add_toolbar_actions(self, toolbar):
        self.action_zoom_in = add_action(toolbar, self,  "Zoom In", on_triggered=self.stock_toolbar.zoom,
                                         icon_name='fa.search-plus', checkable=True)
        self.action_zoom_out = add_action(toolbar, self,  "Zoom Out", on_triggered=self.stock_toolbar.back,
                                          icon_name='fa.search-minus', checkable=False)
        self.action_toggle_legends = add_action(toolbar, self, "Legends", checkable=True,
                                                checked=True)
        toolbar.addSeparator()
        self.action_keep = add_action(toolbar, self,  "Keep", checkable=True)
        self.action_make_current = add_action(toolbar, self,  "Make Current",
                                              checkable=True, checked=True)
        self.keep_make_current_group = QtWidgets.QActionGroup(self)
        self.keep_make_current_group.addAction(self.action_keep)
        self.keep_make_current_group.addAction(self.action_make_current)
        self.keep_make_current_seperator = toolbar.addSeparator()

        self.action_save_image = add_action(toolbar, self, "Save Image", icon_name='fa.save')
        self.action_print_plot = add_action(toolbar, self,  "Print", icon_name='fa.print')
        self.action_plot_options = add_action(toolbar, self, "Plot Options", icon_name='fa.cog')

        toolbar.addSeparator()
        self.action_interactive_cuts = add_action(toolbar, self,  "Interactive Cuts", checkable=True)
        # options for interactive cuts only
        self.action_save_cut = add_action(toolbar, self,  "Save Cut to Workspace")
        self.action_flip_axis = add_action(toolbar, self,  "Flip Integration Axis",
                                           icon_name='fa.retweet')

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


def create_attribute_name(text):
    """Create the name of an action attribute based on the text"""
    return "action_" + text.replace(" ", "_").lower()


def add_action(holder, parent, text, on_triggered=None, icon_name=None,
               checkable=False, checked=False, visible=True):
    """Create a new action based on the given attributes and add it to the given
    holder"""
    action = QtWidgets.QAction(text, parent)
    if icon_name is not None:
        action.setIcon(qta.icon(icon_name))
    action.setCheckable(checkable)
    action.setChecked(checked)
    action.setVisible(visible)
    if on_triggered is not None:
        action.triggered.connect(on_triggered)
    holder.addAction(action)
    return action
