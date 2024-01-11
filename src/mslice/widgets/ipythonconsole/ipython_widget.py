from __future__ import (absolute_import, division,
                        print_function)

import warnings
from mslice.util.mantid import in_mantid

# Ignore Jupyter/IPython deprecation warnings that we can't do anything about
warnings.filterwarnings('ignore', category=DeprecationWarning, module='IPython.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ipykernel.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='jupyter_client.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='qtconsole.*')
del warnings

try:
    # Later versions of Qtconsole are part of Jupyter
    from qtconsole.rich_jupyter_widget import RichJupyterWidget as RichIPythonWidget
    from qtconsole.inprocess import QtInProcessKernelManager
except ImportError:
    from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
    from IPython.qt.inprocess import QtInProcessKernelManager


class IPythonWidget(RichIPythonWidget):
    """ Extends IPython's qt widget to include setting up and in-process kernel
    """

    def __init__(self, *args, **kw):
        super(IPythonWidget, self).__init__(*args, **kw)

        # Create an in-process kernel
        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel = kernel_manager.kernel
        kernel.gui = 'qt'

        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        self.kernel_manager = kernel_manager
        self.kernel_client = kernel_client
        if not in_mantid():
            self.execute('from mslice.util.mantid.mantid_algorithms import *', hidden=True)
            self.execute('from mslice.cli import *', hidden=True)
        else:
            self.execute('import mslice.cli as mc')

    def cleanup(self):
        if in_mantid():
            self.execute('cls')
            self.execute('import mslice.cli as mc')
