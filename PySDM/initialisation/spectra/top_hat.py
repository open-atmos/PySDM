"""
top-hat spectrum
"""

import numpy as np


class TopHat:
    def __init__(self, norm_factor, endpoints):
        self.norm_factor = norm_factor
        self.endpoints = endpoints
        self._mn = endpoints[0]
        self._mx = endpoints[1]

    def cumulative(self, arg):
        cdf = np.minimum(1, np.maximum(0, (arg - self._mn) / (self._mx - self._mn)))
        return self.norm_factor * cdf

    def percentiles(self, cdf_values):
        return (self._mx - self._mn) * (
            np.asarray(cdf_values) + self._mn / (self._mx - self._mn)
        )
