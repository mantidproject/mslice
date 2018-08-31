
def in_mantidplot():
    try:
        import mantidplot  # noqa: F401
    except ImportError:
        return False
    else:
        return True
