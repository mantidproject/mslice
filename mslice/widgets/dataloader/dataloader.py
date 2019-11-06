from __future__ import (absolute_import, division, print_function)

import os

from functools import partial

from mslice.util.qt.QtWidgets import QWidget, QFileSystemModel, QAbstractItemView, QMessageBox
from mslice.util.qt.QtCore import Signal, QDir, Qt

from mslice.presenters.data_loader_presenter import DataLoaderPresenter
from mslice.util.qt import load_ui
from .inputdialog import EfInputDialog

MSLICE_EXTENSIONS = ['*.nxs', '*.nxspe', '*.txt', '*.xye']


class DataLoaderWidget(QWidget):  # and some view interface

    error_occurred = Signal('QString')
    busy = Signal(bool)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        load_ui(__file__, 'dataloader.ui', self)

        self.directory = QDir(os.path.expanduser('~'))
        self._sort_column = 0
        self.reload_model()
        self.table_view.horizontalHeader().swapSections(1, 3)  # Swap the type and date modified columns
        self.txtpath.setText(self.directory.absolutePath())
        self._presenter = DataLoaderPresenter(self)
        self.btnload.setEnabled(False)
        self.btnmerge.setEnabled(False)

        self.table_view.activated.connect(self.activated)
        self.table_view.clicked.connect(self.validate_selection)
        self.txtpath.editingFinished.connect(self.refresh)
        self.btnback.clicked.connect(self.back)
        self.sort.currentIndexChanged.connect(self.sort_files)
        self.btnhome.clicked.connect(self.go_to_home)
        self.btnload.clicked.connect(partial(self.load, False))
        self.btnmerge.clicked.connect(partial(self.load, True))
        self.btnrefresh.clicked.connect(self.reload_model)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace:
            self.back()
        else:
            event.accept()

    def activated(self, file_clicked):
        file_clicked = file_clicked.sibling(file_clicked.row(), 0)  # so clicking anywhere on row gives filename
        if self.file_system.isDir(file_clicked):
            self.enter_dir(self.file_system.fileName(file_clicked))
        else:
            self.load(False)

    def enter_dir(self, directory):
        self.directory.cd(directory)
        self._update_from_path()

    def refresh(self):
        path_entered = QDir(self.txtpath.text())
        if path_entered.exists():
            self.directory = path_entered
            self._update_from_path()
        else:
            self._display_error("Invalid file path")

    def reload_model(self):
        # Redefines the QFileSystemModel - hopefully to refresh any changes not caught because of filesystem issues
        self.file_system = QFileSystemModel()
        self.root_path = self.directory.absolutePath()
        self.file_system.setRootPath(self.root_path)
        self.file_system.setNameFilters(MSLICE_EXTENSIONS)
        self.file_system.setNameFilterDisables(False)
        self.table_view.setModel(self.file_system)
        self.table_view.setRootIndex(self.file_system.index(self.root_path))
        self.table_view.setColumnWidth(0, 320)    # Make name wide
        self.table_view.setColumnWidth(1, 123)
        self.table_view.setColumnWidth(2, 123)
        self.table_view.setColumnWidth(3, 140)    # Show date modified
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sort_files(self._sort_column)

    def _update_from_path(self):
        new_path = self.directory.absolutePath()
        try:
            rel_path = os.path.relpath(new_path, self.root_path)
        except ValueError:   # We are in windows and user changed to another drive
            rel_path = '..'
        if rel_path.startswith('..'):
            self.file_system.setRootPath(new_path)
            self.root_path = new_path
        self.table_view.setRootIndex(self.file_system.index(new_path))
        self.txtpath.setText(new_path)
        self._clear_displayed_error()

    def back(self):
        self.directory.cdUp()
        self._update_from_path()

    def load(self, merge):
        self._presenter.load_workspace(self.get_selected_file_paths(), merge)

    def sort_files(self, column):
        self._sort_column = column
        self.table_view.sortByColumn(column, column % 2)  # descending order for size/modified, ascending for name/type

    def go_to_home(self):
        self.directory = QDir(os.path.expanduser('~'))
        self._update_from_path()

    def validate_selection(self):
        self.btnload.setEnabled(False)
        self.btnmerge.setEnabled(False)
        selected = self.get_selected_file_paths()
        for selection in selected:
            if self.file_system.isDir(self.file_system.index(selection)):
                return
        self.btnload.setEnabled(True)
        if len(selected) > 1:
            self.btnmerge.setEnabled(True)

    def get_selected_file_paths(self):
        selected = self.table_view.selectionModel().selectedRows()
        for i in range(len(selected)):
            selected[i] = selected[i].sibling(selected[i].row(), 0)
            selected[i] = str(os.path.join(self.directory.absolutePath(), self.file_system.fileName(selected[i])))
        return selected

    def get_workspace_efixed(self, workspace, hasMultipleWS=False, default_value=None):
        Ef, applyToAll, success = EfInputDialog.getEf(workspace, hasMultipleWS, default_value)
        if not success:
            raise ValueError('Fixed final energy not given')
        return Ef, applyToAll

    def get_presenter(self):
        return self._presenter

    def error_unable_to_open_file(self, filename=None):
        self._display_error('MSlice was not able to load %s' % ('the selected file' if filename is None else filename))

    def error_merge_different_file_formats(self):
        self._display_error('Cannot merge files with different formats')

    def no_workspace_has_been_loaded(self, filename=None):
        if filename is None:
            self._display_error('No new workspaces have been loaded')
        else:
            self._display_error('File %s has not been loaded' % filename)

    def confirm_overwrite_workspace(self):
        text = 'The workspace you want to load has the same name as an existing workspace,' \
               'Are you sure you want to overwrite it?'
        reply = QMessageBox.question(self,'Confirm Overwrite', text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            return True
        else:
            return False

    def show_busy(self, is_busy):
        self.busy.emit(is_busy)

    def error_loading_workspace(self, message):
        self._display_error(str(message))

    def _display_error(self, error_string):
        self.error_occurred.emit(error_string)

    def _clear_displayed_error(self):
        self._display_error("")
