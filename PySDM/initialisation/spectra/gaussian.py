"""
Gaussian/normal spectrum implemented using
 [SciPy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
"""

from scipy.stats import norm

from PySDM.initialisation.impl.spectrum import Spectrum


class Gaussian(Spectrum):
    def __init__(self, norm_factor, loc, scale):
        super().__init__(norm, (loc, scale), norm_factor)  # mean  # std dev
