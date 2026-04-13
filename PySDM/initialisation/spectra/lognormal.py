"""
lognormal spectrum implemented using
 [SciPy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
"""

import math

from scipy.stats import lognorm

from PySDM.initialisation.impl.spectrum import Spectrum


class Lognormal(Spectrum):
    def __init__(self, *, norm_factor: float, median = None, s_geom: float, mode = None):
        """`norm_factor=1` corresponds to standard normalised probability density,
        other settings allow to express, e.g., size or mass distributions;
        `m_mode` is the median value, `s_geom` is the geometric standard deviation"""
        assert (median is None) ^ (mode is None)
        self.median = median or mode * math.exp(s_geom**2)
        super().__init__(lognorm, (math.log(s_geom), 0, median), norm_factor)

    @property
    def s_geom(self):
        return math.exp(self.distribution_params[0])

    @property
    def m_mode(self):
        raise NotImplementedError()

    @property
    def geometric_mean(self):
        return self.s_geom

    def __str__(self):
        return (
            f"{self.__class__.__name__}:"
            f" (N={self.norm_factor:.3g},"
            f" median={self.median:.3g},"
            f" s_geom={self.s_geom:.3g})"
        )
