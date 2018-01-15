from mslice.util.qt.uic import loadUi, loadUiType
import os.path as osp

def load_ui(caller_filename, ui_relfilename, baseinstance=None):
    """Load a ui file assuming it is relative to a given callers filepath. If
    an existing instance is given then the this instance is set as the baseclass
    and the new Ui instance is returned otherwise the loaded Ui class is returned
    :param caller_filename: An absolute path to a file whose basename when
    joined with ui_relfilename gives the full path to the .ui file. Generally
    this called with __file__
    :param ui_relfilename: A relative filepath that when joined with the
    basename of caller_filename gives the absolute path to the .ui file.
    :param baseinstance: An instance of a widget to pass to uic.loadUi
    that becomes the base class rather than a new widget being created.
    :return: A new instance of the form class if baseinstance is given, otherwise
    return the form class
    """
    filepath = osp.join(osp.dirname(caller_filename), ui_relfilename)
    if baseinstance is not None:
        return loadUi(filepath, baseinstance=baseinstance)
    else:
        return loadUiType(filepath)
