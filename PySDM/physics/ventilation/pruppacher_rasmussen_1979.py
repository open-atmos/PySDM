"""
ventilation coefficient as a function of dimensionless Reynolds (Re) and Schmidt (Sc) numbers for liquid drops
following [Pruppacher & Rasmussen (1979))](https://doi.org/10.1175/1520-0469(1979)036<1255:AWTIOT>2.0.CO;2)

NB: this parameterization is only experimentally validated for Re < 2600 but is hypothesized to be valid for spheres with Re < 8 × 10⁴ based on theory (Pruppacher & Rasmussen, 1979).
the parameterization also does not account for effects of air turbulence.
"""

import numpy as np


class PruppacherRasmussen:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def calcηairEarth(const, T):
        """
        calculate dynamic viscosity of Earth air
        from [Zografos et al. (1987)](doi:10.1016/0045-7825(87)90003-X) Table 1
        (note labeled as μ not η there)
        fit for T ∈ [100-3000] K
        neglects effects of pressure
        input:
            * T [K] - temperature
        output:
            * η [Pa s] - dynamic viscosity
        """
        return (
            const.ZOGRAFOS_1987_COEFF_T3 * np.power(T, const.THREE)
            + const.ZOGRAFOS_1987_COEFF_T2 * np.power(T, const.TWO)
            + const.ZOGRAFOS_1987_COEFF_T1 * T
            + const.ZOGRAFOS_1987_COEFF_T0
        )

    @staticmethod
    def calcRe_diameter(const, radius, v, ηair, ρair):
        """
        calculate particle Reynolds number
        assumes length scale is particle diameter
        inputs:
            * const [dic] - constants dictionary
            * radius [m] - drop radius
            * v [m s⁻¹] - drop velocity wrt air motion
            * ηair [Pa s] - air dynamic viscosity
            * ρair [kg m⁻³] - air density
        output:
            * Re [ ] - Reynolds number
        """
        return const.TWO * radius * v * ρair / ηair

    @staticmethod
    def calcSc(ηair, D, ρair):
        """
        calculate Schmidt number of air
        inputs:
        outputs:
            * D [m² s⁻¹] - diffusivity of H2O (condensible) vapor in air
            * ηair [Pa s] - air dynamic viscosity
            * ρair [kg m⁻³] - air density
        output:
            * Sc [ ] - Schmidt number
        """
        return ηair / D / ρair  # [ ] Schmidt number

    @staticmethod
    def calcX(const, Re, Sc):
        """
        calculate X as defined in [Pruppacher & Rasmussen (1979))](https://doi.org/10.1175/1520-0469(1979)036<1255:AWTIOT>2.0.CO;2)
        inputs:
            * const [dic] - constants dictionary
            * Re [ ] - particle Reynolds number
            * Sc [ ] - air Schmidt number
        return:
            * X [ ] - nondimensional number used to calculate ventilation factor
        """
        return np.power(Re, const.ONE_HALF) * np.power(Sc, const.ONE_THIRD)

    @staticmethod
    def calcfV_small(const, X):
        """
        calculate ventilation coefficient for smaller particles with X < 1.4
        originally derived in [Beard & Pruppacher (1971)](https://doi.org/10.1175/1520-0469(1971)028<1455:awtiot>2.0.co;2)
        inputs:
            * const [dic] - constants dictionary
            * X [ ] - nondimensional number used to calculate ventilation factor
        output:
            * fV [ ] - ventilation coefficent
        """
        return (
            const.PRUPPACHER_RASMUSSEN_1979_CONSTSMALL
            + const.PRUPPACHER_RASMUSSEN_1979_COEFFSMALL
            * np.power(X, const.PRUPPACHER_RASMUSSEN_1979_POWSMALL)
        )

    @staticmethod
    def calcfV_big(const, X):
        """
        calculate ventilation coefficient for bigger particles with X ≥ 1.4
        inputs:
            * const [dic] - constants dictionary
            * X [ ] - nondimensional number used to calculate ventilation factor
        output:
            * fV [ ] - ventilation coefficent
        """
        return (
            const.PRUPPACHER_RASMUSSEN_1979_CONSTBIG
            + const.PRUPPACHER_RASMUSSEN_1979_COEFFBIG * X
        )

    @staticmethod
    def calcfV(const, X):  # this should maybe occur outside here?
        """
        calculate ventliation coefficient for all particles
        inputs:
            * const [dic] - constants dictionary
            * X [ ] - nondimensional number used to calculate ventilation factor
        output:
            * fV [ ] ventilation coefficient
        """
        if X < const.PRUPPACHER_RASMUSSEN_1979_XTHRES:
            return (
                const.PRUPPACHER_RASMUSSEN_1979_CONSTSMALL
                + const.PRUPPACHER_RASMUSSEN_1979_COEFFSMALL
                * np.power(X, const.PRUPPACHER_RASMUSSEN_1979_POWSMALL)
            )
        else:
            return (
                const.PRUPPACHER_RASMUSSEN_1979_CONSTBIG
                + const.PRUPPACHER_RASMUSSEN_1979_COEFFBIG * X
            )
