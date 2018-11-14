from contextlib import contextmanager
from mslice.cli.cli_data_loader import CLIDataLoaderWidget


@contextmanager
def show_busy(view):
    if isinstance(view, CLIDataLoaderWidget):
        yield
        return
    view.busy.emit(True)
    yield
    view.busy.emit(False)
