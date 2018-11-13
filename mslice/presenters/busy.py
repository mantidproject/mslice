from contextlib import contextmanager


@contextmanager
def show_busy(view):
    view.busy.emit(True)
    yield
    view.busy.emit(False)
