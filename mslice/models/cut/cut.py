
class Cut(object):
    """Groups parameters needed to cut and validates them"""

    def __init__(self, cut_axis, integration_axis, intensity_start, intensity_end, norm_to_one=False, width=None):
        self.cut_axis = cut_axis
        self.integration_axis = integration_axis
        self.intensity_start = intensity_start
        self.intensity_end = intensity_end
        self.norm_to_one = norm_to_one
        self.icut = None
        # These are here to save the integration range as the integration axis is
        # changed when a cut with a width is performed
        self.start = integration_axis.start
        self.end = integration_axis.end
        self.width = width

    def reset_integration_axis(self, start, end):
        self.integration_axis.start = start
        self.integration_axis.end = end

    @property
    def cut_axis(self):
        return self._cut_axis

    @cut_axis.setter
    def cut_axis(self, axis):
        self._cut_axis = axis

    @property
    def integration_axis(self):
        return self._integration_axis

    @integration_axis.setter
    def integration_axis(self, axis):
        self._integration_axis = axis

    @property
    def intensity_start(self):
        return self._intensity_start

    @intensity_start.setter
    def intensity_start(self, int_start):
        if int_start is None:
            self._intensity_start = None
        else:
            try:
                self._intensity_start = None if int_start == '' else float(int_start)
            except ValueError:
                raise ValueError('Invalid intensity parameters')

    @property
    def intensity_end(self):
        return self._intensity_end

    @intensity_end.setter
    def intensity_end(self, int_end):
        if int_end is None:
            self._intensity_end = None
        else:
            try:
                self._intensity_end = None if int_end == '' else float(int_end)
            except ValueError:
                raise ValueError('Invalid intensity parameters')

    @property
    def norm_to_one(self):
        return self._norm_to_one

    @norm_to_one.setter
    def norm_to_one(self, value):
        self._norm_to_one = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width_str):
        if width_str is not None and width_str.strip():
            try:
                self._width = float(width_str)
            except ValueError:
                raise ValueError("Invalid width")
        elif width_str == '':
            self._width = self.end - self.start
        else:
            self._width = None
