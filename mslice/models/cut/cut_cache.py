from mslice.presenters.presenter_utility import PresenterUtility


class CutCache(PresenterUtility):
    '''
    Groups parameters needed to cut and validates them
    '''

    def unpack(self):
        return (self.cut_axis, self.integration_axis, self.norm_to_one, self.intensity_start,
                self.intensity_end, self.width)

    @property
    def cut_axis(self):
        return self._cut_axis

    @cut_axis.setter
    def cut_axis(self, axis):
        self._cut_axis = self.validate_axis(axis)

    @property
    def integration_axis(self):
        return self._integration_axis

    @integration_axis.setter
    def integration_axis(self, axis):
        self._integration_axis = self.validate_axis(axis)

    @property
    def intensity_start(self):
        return self._intensity_start

    @intensity_start.setter
    def intensity_start(self, int_start):
        try:
            self._intensity_start = self._to_float(int_start)
        except ValueError:
            raise ValueError('Invalid intensity parameters')

    @property
    def intensity_end(self):
        return self._intensity_end

    @intensity_end.setter
    def intensity_end(self, int_end):
        try:
            self._intensity_end = self._to_float(int_end)
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
        if width_str.strip():
            try:
                self._width = float(width_str)
            except ValueError:
                raise ValueError("Invalid width")
        else:
            self._width = None