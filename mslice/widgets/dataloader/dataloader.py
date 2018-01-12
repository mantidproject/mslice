from __future__ import (absolute_import, division, print_function)

import os

from qtpy.QtWidgets import QWidget, QFileSystemModel, QAbstractItemView
from qtpy.QtCore import QDir

from mslice.util.qt import load_ui


class DataLoaderWidget(QWidget): # and some view interface

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        load_ui(__file__, 'dataloader.ui', self)

        self.file_system = QFileSystemModel()
        self.file_system.setRootPath(QDir.currentPath())
        self.directory = QDir("C:/")
        self.table_view.setModel(self.file_system)
        self.table_view.setColumnWidth(0, 320)
        self.table_view.setColumnWidth(1, 0)
        self.table_view.setColumnWidth(3, 0)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.table_view.doubleClicked.connect(self.clicked)
        self.btnrefresh.clicked.connect(self.refresh)
        self.btnback.clicked.connect(self.back)
        self.btnload.clicked.connect(self.load)

    def clicked(self, file_clicked):
        file_clicked = file_clicked.sibling(file_clicked.row(), 0) # so clicking anywhere on row gives filename
        self.directory.cd(self.file_system.fileName(file_clicked))
        self._update_from_path()

    def refresh(self): #TODO: needs error checking. Also, perform automatically on line edit?
        self.directory = QDir(self.txtpath.text())
        self._update_from_path()

    def _update_from_path(self):
        new_path = self.directory.absolutePath()
        self.table_view.setRootIndex(self.file_system.index(new_path))
        self.txtpath.setText(new_path)

    def back(self):
        self.directory.cdUp()
        self._update_from_path()

    def load(self):
        file_selected = self.table_view.selectionModel().selectedRows()[0] #TODO: if multiple selected?
        file_selected = file_selected.sibling(file_selected.row(), 0)
        file_selected = os.path.join(self.directory.absolutePath(), self.file_system.fileName(file_selected))
        print(file_selected)
        self.presenter.load_workspace(file_selected)

    def get_presenter(self):
        return self._presenter
