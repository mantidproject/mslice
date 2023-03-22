from mslice.models.units import EnergyUnits


class Axis(object):
    def __init__(self, units, start, end, step, e_unit='meV'):
        self.units = units
        self.start = float(start)
        self.end = float(end)
        self.step = float(step)
        self.e_unit = e_unit

    def to_dict(self):
        return {'start': self.start, 'end': self.end, 'step': self.step, 'units': self.units, 'e_unit': self.e_unit}

    def __str__(self):
        return "{},{},{},{}{}".format(self.units, self.start, self.end, self.step,
                                      ",{}".format(self.e_unit) if self.not_meV else "")

    def __eq__(self, other):
        # This is required for Unit testing
        return self.units == other.units and self.start == other.start and self.end == other.end \
            and self.step == other.step and isinstance(other, Axis)

    def __repr__(self):
        info = (self.units, self.start, self.end, self.step)
        if self.not_meV:
            info += (self.e_unit,)
        return "Axis(" + " ,".join(map(repr, info)) + ")"

    @property
    def not_meV(self):
        return "DeltaE" in self.units and "meV" not in self.e_unit

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        try:
            self._start = float(value)
        except ValueError:
            if str(value) == '':
                raise ValueError("Invalid axis parameter on {}: Start value required!".format(self.units))
            else:
                raise ValueError("Invalid axis parameter on {}: "
                                 "Start value {} is not a valid float!".format(self.units, value))

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        try:
            end_float = float(value)
        except ValueError:
            if str(value) == '':
                raise(ValueError("Invalid axis parameter on {}: End value required!".format(self.units)))
            else:
                raise ValueError("Invalid axis parameter on {}: "
                                 "End value {} is not a valid float!".format(self.units, value))
        if end_float <= self.start:
            raise ValueError("Invalid axis parameter on {}: End value must be greater"
                             " than start value!".format(self.units))
        self._end = end_float

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        try:
            self._step = float(value)
        except ValueError:
            raise ValueError("Invalid axis parameter on {}: Step {} is not a valid float!".format(self.units, value))

    @property
    def start_meV(self):
        return self._start * self.scale

    @property
    def end_meV(self):
        return self._end * self.scale

    @property
    def step_meV(self):
        return self._step * self.scale

    @property
    def e_unit(self):
        try:
            return self._e_unit
        except AttributeError:
            return None

    @e_unit.setter
    def e_unit(self, value):
        old_e_unit = self.e_unit
        self._e_unit = str(value).strip()
        new_e_unit = EnergyUnits(self._e_unit)
        self.scale = new_e_unit.factor_to_meV() if ('DeltaE' in self.units) else 1.
        if old_e_unit is not None and 'DeltaE' in self.units and old_e_unit != self._e_unit:
            # This means the axes had a different previous e_unit set so (start, end, step) should be rescaled
            self.start, self.step, self.end = new_e_unit.convert_from(old_e_unit, self.start, self.step, self.end)
