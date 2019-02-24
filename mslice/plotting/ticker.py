"""
Custom tick formater and locator for energy units "conversion" (scales original axes by a factor)
"""

import math
import numpy as np
from matplotlib import ticker


class ScaledScalarFormatter(ticker.ScalarFormatter):
    def __init__(self, scale=8.066, useOffset=None, useMathText=None, useLocale=None):
        self.scalefactor = scale
        ticker.ScalarFormatter.__init__(self, useOffset, useMathText, useLocale)

    def __call__(self, x, pos=None):
        'Return the format for tick val *x* at position *pos*'
        if len(self.locs) == 0:
            return ''
        else:
            s = self.pprint_val(x * self.scalefactor)
            return self.fix_minus(s)

    def _set_format(self, vmin, vmax):
        # set the format string to format all the ticklabels
        if len(self.locs) < 2:
            # Temporarily augment the locations with the axis end points.
            _locs = list(self.locs) + [vmin * self.scalefactor, vmax * self.scalefactor]
        else:
            _locs = self.locs * self.scalefactor
        locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
        loc_range = np.ptp(locs)
        # Curvilinear coordinates can yield two identical points.
        if loc_range == 0:
            loc_range = np.max(np.abs(locs))
        # Both points might be zero.
        if loc_range == 0:
            loc_range = 1
        if len(self.locs) < 2:
            # We needed the end points only for the loc_range calculation.
            locs = locs[:-2]
        loc_range_oom = int(math.floor(math.log10(loc_range)))
        # first estimate:
        sigfigs = max(0, 3 - loc_range_oom)
        # refined estimate:
        thresh = 1e-3 * 10 ** loc_range_oom
        while sigfigs >= 0:
            if np.abs(locs - np.round(locs, decimals=sigfigs)).max() < thresh:
                sigfigs -= 1
            else:
                break
        sigfigs += 1
        self.format = '%1.' + str(sigfigs) + 'f'
        if self._usetex:
            self.format = '$%s$' % self.format
        elif self._useMathText:
            self.format = '$\mathdefault{%s}$' % self.format

    def format_data(self, value):
        value *= self.scalefactor
        b = self.labelOnlyBase
        self.labelOnlyBase = False
        value = cbook.strip_math(self.__call__(value))
        self.labelOnlyBase = b
        return value

    def format_data_short(self, value):
        return '%-12g' % (value * self.scalefactor)

    @property
    def scalefactor(self):
        return self._scale

    @scalefactor.setter
    def scalefactor(self, val):
        self._scale = float(val)


class ScaledAutoLocator(ticker.MaxNLocator):
    def __init__(self, scale=8.066):
        self.scalefactor = scale
        ticker.MaxNLocator.__init__(self, nbins=9, steps=[1, 2, 5, 10])

    def bin_boundaries(self, vmin, vmax):
        nbins = self._nbins
        scale, offset = ticker.scale_range(vmin * self.scalefactor, vmax * self.scalefactor, nbins)
        if self._integer:
            scale = max(1, scale)
        vmin = (vmin * self.scalefactor - offset)
        vmax = (vmax * self.scalefactor - offset)
        raw_step = (vmax - vmin) / nbins
        scaled_raw_step = raw_step / scale
        best_vmax = vmax
        best_vmin = vmin

        for step in self._steps:
            if step < scaled_raw_step:
                continue
            step *= scale
            best_vmin = step * divmod(vmin, step)[0]
            best_vmax = best_vmin + step * nbins
            if (best_vmax >= vmax):
                break
        if self._trim:
            extra_bins = int(divmod((best_vmax - vmax), step)[0])
            nbins -= extra_bins
        return (np.arange(nbins + 1) * step + best_vmin + offset) / self.scalefactor

    @property
    def scalefactor(self):
        return self._scale

    @scalefactor.setter
    def scalefactor(self, val):
        self._scale = float(val)
