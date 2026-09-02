import sys
import types
import unittest
from unittest import mock

from qtpy.QtWidgets import QApplication
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget

from mslice.widgets.ipythonconsole import ipython_widget


def _make_app():
    return QApplication.instance() or QApplication(sys.argv)


class _FakeWorkbenchConsole(RichJupyterWidget):
    """Stand-in for mantidqt.widgets.jupyterconsole.InProcessJupyterConsole,
    built the same way: it owns its own QtInProcessKernelManager."""

    def __init__(self):
        super().__init__()
        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = "qt"
        kernel_client = kernel_manager.client()
        kernel_client.start_channels()
        self.kernel_manager = kernel_manager
        self.kernel_client = kernel_client


class IPythonWidgetTest(unittest.TestCase):
    """Regression tests for mslice#1235: MSlice's console must not steal
    Mantid Workbench's in-process Jupyter kernel from underneath it.

    Test methods are numbered and run in a single, deliberate order: an
    ipykernel in-process kernel's IPython shell is process-wide global state
    (traitlets SingletonConfigurable), so whether a fake Workbench console
    already exists changes the behaviour under test. Numbering keeps that
    ordering explicit rather than relying on alphabetical accident.

    mantidqt.widgets.jupyterconsole.InProcessJupyterConsole is stubbed into
    sys.modules once for the whole class (not per-test) so real MSlice
    console/kernel objects can be created and torn down without repeatedly
    mutating sys.modules mid-run.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = _make_app()
        cls._fake_module = types.ModuleType("mantidqt.widgets.jupyterconsole")
        cls._fake_module.InProcessJupyterConsole = _FakeWorkbenchConsole
        cls._sys_modules_patcher = mock.patch.dict(
            sys.modules,
            {
                "mantidqt": types.ModuleType("mantidqt"),
                "mantidqt.widgets": types.ModuleType("mantidqt.widgets"),
                "mantidqt.widgets.jupyterconsole": cls._fake_module,
            },
        )
        cls._sys_modules_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls._sys_modules_patcher.stop()

    def test_1_no_workbench_console_falls_back_to_its_own_kernel(self):
        # No Workbench console exists yet in this process.
        self.assertIsNone(ipython_widget.find_workbench_kernel_manager())

        with mock.patch.object(ipython_widget, "in_mantid", return_value=False):
            standalone_widget = ipython_widget.IPythonWidget()
        self.assertTrue(standalone_widget._owns_kernel)
        self.assertIsNotNone(standalone_widget.kernel_manager.kernel)
        standalone_widget.cleanup()

        with mock.patch.object(ipython_widget, "in_mantid", return_value=True):
            embedded_widget = ipython_widget.IPythonWidget()
        self.assertTrue(embedded_widget._owns_kernel)
        embedded_widget.cleanup()

    def test_2_with_workbench_console_reuses_and_detaches_cleanly(self):
        workbench_console = _FakeWorkbenchConsole()
        workbench_kernel = workbench_console.kernel_manager.kernel

        self.assertIs(
            ipython_widget.find_workbench_kernel_manager(),
            workbench_console.kernel_manager,
        )

        with mock.patch.object(ipython_widget, "in_mantid", return_value=True):
            mslice_console = ipython_widget.IPythonWidget()

        # No competing kernel should have been created ...
        self.assertFalse(mslice_console._owns_kernel)
        self.assertIs(mslice_console.kernel_manager, workbench_console.kernel_manager)
        self.assertIs(mslice_console.kernel_manager.kernel, workbench_kernel)

        # ... and MSlice's console is registered as a second frontend of the
        # one shared kernel, not a replacement for it.
        self.assertEqual(len(workbench_kernel.frontends), 2)

        mslice_console.cleanup()

        # Closing MSlice's console must leave Workbench's kernel, its shell,
        # and its own frontend registration completely intact - this is the
        # exact failure reported in mslice#1235.
        self.assertIs(workbench_console.kernel_manager.kernel, workbench_kernel)
        self.assertEqual(len(workbench_kernel.frontends), 1)
        self.assertIs(
            workbench_kernel.shell.display_pub.pub_socket,
            workbench_kernel.iopub_socket,
        )

        workbench_console.kernel_manager.shutdown_kernel()


if __name__ == "__main__":
    unittest.main()
