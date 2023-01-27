import matplotlib as mpl
import matplotlib._pylab_helpers as _pylab_helpers
from qtpy import QT_VERSION

from mslice.plotting.globalfiguremanager import CATEGORY_CUT, CATEGORY_SLICE, GlobalFigureManager, set_category


SLICE_METHOD = ['contour', 'contourf', 'hexbin', 'scatter', 'pcolor', 'pcolormesh', 'hist2d', 'imshow', 'quiver', 'specgram', 'streamplot',
                'tricontour', 'tricontourf', 'tripcolor']
CUT_METHOD = ['acorr', 'angle_spectrum', 'arrow', 'axhline', 'axhspan', 'axline', 'axvline', 'axvspan', 'bar', 'barbs', 'barh',
              'bar_label', 'boxplot', 'broken_barh', 'clabel', 'cohere', 'csd', 'errorbar', 'eventplot', 'fill', 'fill_between',
              'fill_betweenx', 'hist', 'stairs', 'hlines', 'loglog', 'magnitude_spectrum', 'phase_spectrum', 'pie', 'plot',
              'plot_date', 'psd', 'quiverkey', 'semilogx', 'semilogy', 'stackplot', 'stem', 'step', 'triplot', 'violinplot', 'vlines',
              'xcorr']


def init_mpl_gcf():
    """
    Replace vanilla Gcf with our custom manager
    """
    setattr(_pylab_helpers, "Gcf", GlobalFigureManager)


def initialize_matplotlib():
    mpl.use('Qt{}Agg'.format(QT_VERSION[0]))
    add_decorators()


def add_decorators():
    for method in SLICE_METHOD:
        method = set_category(CATEGORY_SLICE)(method)
    for method in CUT_METHOD:
        method = set_category(CATEGORY_CUT)(method)
