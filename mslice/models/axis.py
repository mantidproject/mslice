class Axis(object):
    def __init__(self, units, start, end, step):
        self.units = units
        self.start = start
        self.end = end
        self.step = step

    def to_dict(self):
        return {'start': self.start, 'end': self.end, 'step': self.step, 'units': self.units}

    def __eq__(self, other):
        # This is required for Unit testing
        return self.units == other.units and self.start == other.start and self.end == other.end \
            and self.step == other.step and isinstance(other, Axis)

    def __repr__(self):
        info = (self.units, self.start, self.end, self.step)
        return "Axis(" + " ,".join(map(repr, info)) + ")"

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        try:
            self._start = float(value)
        except ValueError:
            raise ValueError("Invalid axis parameter - start '%s' is not a valid float" % value)

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        try:
            self._end = float(value)
        except ValueError:
            raise ValueError("Invalid axis parameter - end '%s' is not a valid float" % value)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        try:
            self._step = float(value)
        except ValueError:
            raise ValueError("Invalid axis parameter - step '%s' is not a valid float" % value)
