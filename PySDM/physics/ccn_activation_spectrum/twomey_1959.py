import numpy as np
"""
[Twomey 1959](https://doi.org/10.1007/BF01993560). Note that supersaturation is expressed in temperature unit as the elevation of the dew point or as percent of supersaturation; and concentrations are reported for 10C and 800 mb. 
"""

class Twomey1959:
    def __init__(self, const):
        assert np.isfinite(const.TWOMEY_K)
        assert np.isfinite(const.TWOMEY_N0)

    @staticmethod
    def ccn_concentration(const, saturation_ratio):
        return const.TWOMEY_N0*np.power(saturation_ratio-1, const.TWOMEY_K)
    
    
