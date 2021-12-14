from scipy.stats import norm
from PySDM.initialisation.impl.spectrum import Spectrum

class Gaussian(Spectrum):

    def __init__(self, norm_factor, loc, scale):
        super().__init__(norm, (
            loc,     # mean
            scale    # std dev
        ), norm_factor)
