import numpy as np


class KolmogorovInertialRangeScaled:
    def __init__(self, const):
        assert np.isfinite(const.TURBULENT_RELAXATION_TIMESCALE_MULTIPLIER)

    @staticmethod
    def tau(
        const, linear_eddy_length_scale, tke_dissipation_rate
    ):  # pylint: disable=unused-argument
        """eqs 15 and 17 in [Abade and Albuquerque 2024](https://doi.org/10.1002/qj.4775)"""
        return (
            const.TURBULENT_RELAXATION_TIMESCALE_MULTIPLIER
            * tke_dissipation_rate**-const.ONE_THIRD
            * linear_eddy_length_scale**const.TWO_THIRDS
        )
