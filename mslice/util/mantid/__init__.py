
def in_mantid():
    try:
        from mantidqt.gui_helper import get_qapplication
        _, is_mantid = get_qapplication()
        return is_mantid
    except ImportError:
        return False

def in_mantidplot():
    try:
        import mantidplot  # noqa: F401
    except ImportError:
        return False
    else:
        return True
