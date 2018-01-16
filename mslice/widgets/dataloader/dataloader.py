from __future__ import (absolute_import, division, print_function)

import os

from functools import partial

from qtpy.QtWidgets import QWidget, QFileSystemModel, QAbstractItemView, QMessageBox
from qtpy.QtCore import Signal, QDir

from mslice.presenters.data_loader_presenter import DataLoaderPresenter
from mslice.util.qt import load_ui
from .inputdialog import EfInputDialog


class DataLoaderWidget(QWidget): # and some view interface

    error_occurred = Signal('QString')

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        load_ui(__file__, 'dataloader.ui', self)

        self.file_system = QFileSystemModel()
        self.directory = QDir("C:/")
        self.file_system.setRootPath(self.directory.absolutePath())
        self.table_view.setModel(self.file_system)
        self.table_view.setColumnWidth(0, 320)
        self.table_view.setColumnWidth(1, 0)
        self.table_view.setColumnWidth(3, 0)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._presenter = DataLoaderPresenter(self)

        self.table_view.doubleClicked.connect(self.clicked)
        self.btnrefresh.clicked.connect(self.refresh)
        self.txtpath.editingFinished.connect(self.refresh)
        self.btnback.clicked.connect(self.back)
        self.btnload.clicked.connect(partial(self.load, False))
        self.btnmerge.clicked.connect(partial(self.load, True))

    def clicked(self, file_clicked):
        file_clicked = file_clicked.sibling(file_clicked.row(), 0) # so clicking anywhere on row gives filename
        self.directory.cd(self.file_system.fileName(file_clicked))
        self._update_from_path()

    def refresh(self):
        path_entered = QDir(self.txtpath.text())
        if path_entered.exists():
            self.directory = path_entered
            self._update_from_path()
        else:
            self._display_error("Invalid file path")

    def _update_from_path(self):
        new_path = self.directory.absolutePath()
        self.table_view.setRootIndex(self.file_system.index(new_path))
        self.txtpath.setText(new_path)

    def back(self):
        self.directory.cdUp()
        self._update_from_path()

    def load(self, merge):
        self._presenter.load_workspace(self.get_selected_file_paths(), merge)

    def get_selected_file_paths(self):
        selected = self.table_view.selectionModel().selectedRows()
        for i in range(len(selected)):
            selected[i] = selected[i].sibling(selected[i].row(), 0)
            selected[i] = str(os.path.join(self.directory.absolutePath(), self.file_system.fileName(selected[i])))
        return selected

    def get_workspace_efixed(self, workspace, hasMultipleWS=False):
        Ef, applyToAll, success = EfInputDialog.getEf(workspace, hasMultipleWS, None)
        if not success:
            raise ValueError('Fixed final energy not given')
        return Ef, applyToAll

    def get_presenter(self):
        return self._presenter

    def error_unable_to_open_file(self, filename=None):
        self._display_error('MSlice was not able to load %s' % ('the selected file' if filename is None else filename))

    def no_workspace_has_been_loaded(self, filename=None):
        if filename is None:
            self._display_error('No new workspaces have been loaded')
        else:
            self._display_error('File %s has not been loaded' % (filename))

    def confirm_overwrite_workspace(self):
        reply = QMessageBox.question(self,'Confirm Overwrite', 'The workspace you want to load has the same name as'
                                                               'an existing workspace, Are you sure you want to '
                                                               'overwrite it? ',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            return True
        else:
            return False

    def error_loading_workspace(self, message):
        self._display_error(str(message))

    def _display_error(self, error_string):
        self.error_occurred.emit(error_string)
