from contextlib import contextmanager


@contextmanager
def show_busy(view):
    from mslice.cli.cli_helper_classes.cli_data_loader import CLIDataLoaderWidget
    if isinstance(view, CLIDataLoaderWidget):
        yield
        return
    view.busy.emit(True)
    yield
    view.busy.emit(False)
