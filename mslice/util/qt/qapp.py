# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
#  This file is part of the mantid workbench.
#
from __future__ import (absolute_import, unicode_literals)
from functools import wraps
import sys

from mslice.util.qt.QtCore import Qt, QMetaObject, QObject, QThread, Slot
from mslice.util.qt.QtWidgets import QApplication

from six import reraise


qApp = None


class QAppThreadCall(QObject):
    """
    Wraps a callable object and forces any calls made to it to be executed
    on the same thread as the qApp object. This is required for anything
    called by the matplotlib figures, which run on a separate thread.
    """

    def __init__(self, callee):
        global qApp
        create_qapp_if_required()
        super(QAppThreadCall, self).__init__()
        self.moveToThread(qApp.thread())
        self.callee = callee
        # Help should then give the correct doc
        self.__call__.__func__.__doc__ = callee.__doc__
        self._args = None
        self._kwargs = None
        self._result = None
        self._exc_info = None

    def __call__(self, *args, **kwargs):
        """
        If the current thread is the qApp thread then this
        performs a straight call to the wrapped callable_obj. Otherwise
        it invokes the do_call method as a slot via a
        BlockingQueuedConnection.
        """
        global qApp
        if QThread.currentThread() == qApp.thread():
            return self.callee(*args, **kwargs)
        else:
            self._store_function_args(*args, **kwargs)
            QMetaObject.invokeMethod(self, "on_call",
                                     Qt.BlockingQueuedConnection)
            if self._exc_info is not None:
                reraise(*self._exc_info)
            return self._result

    @Slot()
    def on_call(self):
        """Perform a call to a GUI function across a
        thread and return the result
        """
        try:
            self._result = \
                self.callee(*self._args, **self._kwargs)
        except Exception: # pylint: disable=broad-except
            self._exc_info = sys.exc_info()

    def _store_function_args(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        # Reset return value and exception
        self._result = None
        self._exc_info = None


def call_in_qapp_thread(func):
    """
    Decorator to force a call onto the QApplication thread
    :param func: The function to decorate
    :return The wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return QAppThreadCall(func)(*args, **kwargs)

    return wrapper

def create_qapp_if_required():
    """
    Create a qapplication for non gui plots
    """
    global qApp

    if qApp is None:
        instance = QApplication.instance()
        if instance is None:
            instance = QApplication(['mslice'])
            instance.lastWindowClosed.connect(instance.quit)
        qApp = instance
    return qApp


def mainloop():
    global qApp
    qApp.exec_()
