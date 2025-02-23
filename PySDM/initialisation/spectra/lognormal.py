"""
lognormal spectrum implemented using
 [SciPy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
"""

import math

from scipy.stats import lognorm

from PySDM.initialisation.impl.spectrum import Spectrum


class Lognormal(Spectrum):
    def __init__(self, norm_factor: float, m_mode: float, s_geom: float):
        """`norm_factor=1` corresponds to standard normalised probability density,
        other settings allow to express, e.g., size or mass distributions;
        `m_mode` is the median value, `s_geom` is the geometric standard deviation"""
        super().__init__(lognorm, (math.log(s_geom), 0, m_mode), norm_factor)

    @property
    def s_geom(self):
        return math.exp(self.distribution_params[0])

    @property
    def m_mode(self):
        return self.distribution_params[2]

    @property
    def median(self):
        return self.m_mode

    @property
    def geometric_mean(self):
        return self.s_geom

    def __str__(self):
        return (
            f"{self.__class__.__name__}:"
            f" (N={self.norm_factor:.3g},"
            f" m_mode={self.m_mode:.3g},"
            f" s_geom={self.s_geom:.3g})"
        )
