
class CommonWorkspaceProperties:
    """Class to ensure consistency of mslice specific workspace properties across mslice
    to allow for error free propagation. Not necessarily implemented for all workspace types"""

    def __init__(self):
        self.is_PSD = None
        self.parent = None
        self.intensity_corrected = None
        self.axes = None
        self.ef_defined = None
        self.e_mode = None
        self.limits = None
        self.e_fixed = None
