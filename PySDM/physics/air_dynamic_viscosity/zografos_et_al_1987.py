"""
calculate dynamic viscosity of Earth air
from [Zografos et al. (1987)](doi:10.1016/0045-7825(87)90003-X) Table 1
(note labeled as μ not η there)
fit for T ∈ [100-3000] K
neglects effects of pressure
"""


class ZografosEtAl1987:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def eta_air(const, temperature):
        return (
            const.ZOGRAFOS_1987_COEFF_T3 * temperature**3
            + const.ZOGRAFOS_1987_COEFF_T2 * temperature**2
            + const.ZOGRAFOS_1987_COEFF_T1 * temperature
            + const.ZOGRAFOS_1987_COEFF_T0
        )
