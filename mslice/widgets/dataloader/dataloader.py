
from qtpy.QtWidgets import QWidget, QFileSystemModel, QTableView, QGridLayout, QAbstractItemView
from qtpy.QtCore import QDir

from mslice.util.qt import load_ui

class DataLoaderWidget(QWidget): # and some view interface

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        load_ui(__file__, 'dataloader.ui', self)
        self.file_system = QFileSystemModel()
        self.file_system.setRootPath(QDir.currentPath())
        self.table_view.setModel(self.file_system)
        self.table_view.setColumnWidth(0, 320)
        self.table_view.setColumnWidth(3, 150)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.doubleClicked.connect(self.clicked)
        self.table_view.show()


    def clicked(self, file_clicked):
        file_clicked = file_clicked.sibling(file_clicked.row(), 0)
        if self.file_system.isDir(file_clicked):
            self.table_view.setRootIndex(file_clicked)
            self.txtpath.setText(self.file_system.filePath(file_clicked))
