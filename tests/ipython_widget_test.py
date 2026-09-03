import sys
import types
import unittest
from unittest import mock

from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtpy.QtWidgets import QApplication

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

        cls._started_channels_patcher = mock.patch.object(
            RichJupyterWidget, "_started_channels", lambda self: None
        )
        cls._started_channels_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls._started_channels_patcher.stop()
        cls._sys_modules_patcher.stop()

    def _assert_cleanup_only_reimports_cli(self, widget):
        """Assert cleanup() only ever executes the
        "import mslice.cli as mc" re-import."""
        with mock.patch.object(type(widget), "execute") as execute:
            widget.cleanup()
        calls = [
            call.args[0] if call.args else call.kwargs.get("source")
            for call in execute.call_args_list
        ]
        self.assertNotIn("cls", calls)
        self.assertIn("import mslice.cli as mc", calls)

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
            self._assert_cleanup_only_reimports_cli(embedded_widget)

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
            self.assertIs(
                mslice_console.kernel_manager, workbench_console.kernel_manager
            )
            self.assertIs(mslice_console.kernel_manager.kernel, workbench_kernel)

            # ... and MSlice's console is registered as a second frontend of
            # the one shared kernel, not a replacement for it.
            self.assertEqual(len(workbench_kernel.frontends), 2)

            self._assert_cleanup_only_reimports_cli(mslice_console)

        # Closing MSlice's console must leave Workbench's kernel, its shell,
        # and its own frontend registration completely intact.
        self.assertIs(workbench_console.kernel_manager.kernel, workbench_kernel)
        self.assertEqual(len(workbench_kernel.frontends), 1)
        self.assertIs(
            workbench_kernel.shell.display_pub.pub_socket,
            workbench_kernel.iopub_socket,
        )

        workbench_console.kernel_manager.shutdown_kernel()


if __name__ == "__main__":
    unittest.main()
