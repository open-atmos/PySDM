"""
exponential spectrum implemented using
 [SciPy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
"""

from scipy.stats import expon

from PySDM.initialisation.impl.spectrum import Spectrum


class Exponential(Spectrum):
    def __init__(self, norm_factor, scale):
        super().__init__(expon, (0, scale), norm_factor)  # loc  # scale = 1/lambda
