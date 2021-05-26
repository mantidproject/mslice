
def in_mantid():
    try:
        from mantidqt.gui_helper import get_qapplication
        _, is_mantid = get_qapplication()
        return is_mantid
    except ImportError:
        return False
