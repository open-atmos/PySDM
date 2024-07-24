"""
Eq. 1 in [Kaul et al. 2015](https://doi.org/10.1175/MWR-D-14-00319.1)
"""

import numpy as np


class KaulEtAl2015:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.bulk_phase_partitioning_exponent)

    @staticmethod
    def liquid_fraction(const, T):
        return np.minimum(
            1,
            np.power(
                np.maximum(
                    0,
                    (T - const.bulk_phase_partitioning_T_cold)
                    / (
                        const.bulk_phase_partitioning_T_warm
                        - const.bulk_phase_partitioning_T_cold
                    ),
                ),
                const.bulk_phase_partitioning_exponent,
            ),
        )
