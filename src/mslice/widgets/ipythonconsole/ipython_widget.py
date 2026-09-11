import warnings

from mslice.util.mantid import in_mantid

# Ignore Jupyter/IPython deprecation warnings that we can't do anything about
warnings.filterwarnings("ignore", category=DeprecationWarning, module="IPython.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ipykernel.*")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="jupyter_client.*"
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qtconsole.*")
del warnings

try:
    # Later versions of Qtconsole are part of Jupyter
    from qtconsole.inprocess import QtInProcessKernelManager
    from qtconsole.rich_jupyter_widget import RichJupyterWidget as RichIPythonWidget
except ImportError:
    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
    from IPython.qt.inprocess import QtInProcessKernelManager


def find_workbench_kernel_manager():
    """Return the kernel manager of Mantid Workbench's own IPython console dock,
    if one is already running in this process, otherwise None.

    """
    try:
        from mantidqt.widgets.jupyterconsole import InProcessJupyterConsole
        from qtpy.QtWidgets import QApplication
    except ImportError:
        return None

    app = QApplication.instance()
    if app is None:
        return None

    for widget in app.allWidgets():
        if isinstance(widget, InProcessJupyterConsole):
            return widget.kernel_manager
    return None


class IPythonWidget(RichIPythonWidget):
    """Extends IPython's qt widget to include setting up and in-process kernel"""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._connect_kernel()
        self._import_mslice_namespace()

    def _connect_kernel(self):
        """Reuse Mantid Workbench's own console kernel if one is already running
        in this process, rather than starting a competing in-process kernel
        that would silently break it. Only fall back to creating our own
        kernel when running standalone, or if Workbench's console cannot be
        found (see find_workbench_kernel_manager for why this matters).

        Besides __init__, this is also called by resume() to reconnect after
        cleanup() has detached this console - e.g. when MSlice's window is
        closed and reopened, the same window (and this console) is reused
        rather than recreated, see MainWindow.
        """
        kernel_manager = find_workbench_kernel_manager()
        self._owns_kernel = kernel_manager is None
        if kernel_manager is None:
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel()
            kernel = kernel_manager.kernel
            kernel.gui = "qt"

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        self.kernel_manager = kernel_manager
        self.kernel_client = kernel_client
        self._connected = True

    def _import_mslice_namespace(self):
        """Populate this console's namespace with mslice's API - used on
        first connection and again by resume() after a reconnect, since a
        standalone console's new kernel starts with an empty namespace."""
        if not in_mantid():
            self.execute(
                "from mslice.util.mantid.mantid_algorithms import *", hidden=True
            )
            self.execute("from mslice.cli import *", hidden=True)
        else:
            self.execute("import mslice.cli as mc")

    def resume(self):
        """Reconnect this console to a kernel after cleanup() has detached it.

        MSlice's main window is reused rather than recreated when it is
        closed and reopened (see mainwindow.MainWindow), so without this the
        console is left permanently detached from its kernel after the first
        close, and stops showing any output. Safe to call even if this
        console is already connected - it is then a no-op.
        """
        if self._connected:
            return
        self._connect_kernel()
        self._import_mslice_namespace()

    def cleanup(self):
        if in_mantid():
            self.execute("import mslice.cli as mc")

        # Detach this console's client from the kernel it was using. If it
        # shares Workbench's own kernel, this leaves Workbench's console
        # running unaffected - only this widget's own frontend registration
        # is removed. There is no need to explicitly shut down a kernel this
        # console owns either: standalone, it dies with the process when the
        # QApplication exits; embedded in Workbench, Workbench's own
        # QApplication will clean it up when it eventually exits.
        if self.kernel_client is not None:
            self.kernel_client.stop_channels()
            kernel = getattr(self.kernel_manager, "kernel", None)
            if kernel is not None and self.kernel_client in kernel.frontends:
                kernel.frontends.remove(self.kernel_client)
        self._connected = False
