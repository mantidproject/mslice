from contextlib import contextmanager


@contextmanager
def show_busy(view):
    view.show_busy(True)
    yield
    view.show_busy(False)
