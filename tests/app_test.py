from qtpy import QtWidgets
from qtpy.QtCore import Qt, QTimer
import unittest
from mock import patch
from mslice.app.mainwindow import MainWindow
from mslice.plotting.globalfiguremanager import GlobalFigureManager

qapp = QtWidgets.QApplication.instance()


class AppTests(unittest.TestCase):
    def setUp(self):
        global qapp
        if qapp is None:
            qapp = QtWidgets.QApplication([" "])

    def tearDown(self):
        GlobalFigureManager.destroy_all()

        # Required to sendPostedEvents twice to ensure the MainWindow is deleted
        qapp.sendPostedEvents()
        qapp.sendPostedEvents()

        # There should be no widgets hanging around after the MainWindow is closed
        self.assertEqual(0, len(QtWidgets.QApplication.topLevelWidgets()))

    def test_mainwindow(self):
        """Test the MainWindow initialises correctly"""
        with patch.object(MainWindow, "setup_ipython"):
            window = MainWindow()
            window.setAttribute(Qt.WA_DeleteOnClose, True)
            QTimer.singleShot(0, window.close)
