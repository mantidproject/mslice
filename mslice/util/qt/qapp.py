# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
#
from __future__ import (absolute_import, unicode_literals)
from functools import wraps
from typing import Any, Sequence

# make these available in this module for the rest of codebase
from mantidqt.utils.qt.qappthreadcall import QAppThreadCall, force_method_calls_to_qapp_thread  # noqa: F401

from qtpy.QtWidgets import QApplication

# Global QApplication instance reference to keep it alive
qApp = None


def call_in_qapp_thread(func):
    """
    Method decorator to force a call onto the QApplication thread
    :param func: The function to decorate
    :return The wrapped function
    """
    @wraps(func)
    def wrapper(*args: Sequence) -> Any:
        return QAppThreadCall(func)(*args)

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
