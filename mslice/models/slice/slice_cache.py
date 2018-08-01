
class SliceCache():
    '''class that caches intensities and metadata for a single workspace'''
    def __init__(self, slice, colourmap, norm, sample_temp, q_axis, e_axis, rotated=False):
        self.scattering_function = slice
        self.colourmap = colourmap
        self.norm = norm
        self.sample_temp = sample_temp
        self.momentum_axis = q_axis
        self.energy_axis = e_axis
        self.rotated = rotated
        self.overplot_lines = {}

        #intensities
        self.chi = None
        self.chi_magnetic = None
        self.d2sigma = None
        self.symmetrised = None
        self.gdos = None

