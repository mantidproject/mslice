from mslice.util.qt import QtWidgets
import unittest

from mslice.app.mainwindow import MainWindow


qapp = None


class AppTests(unittest.TestCase):

    def setUp(self):
        global qapp
        if qapp is None:
            qapp = QtWidgets.QApplication([' '])

    def test_mainwindow(self):
        """Test the MainWindow initialises correctly"""
        MainWindow()
