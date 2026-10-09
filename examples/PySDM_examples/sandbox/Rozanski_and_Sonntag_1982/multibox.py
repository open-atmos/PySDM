"""
definition of a multi-box environment (parcel iteratively adjusting to a stationary state)
plus helper equation for isotopic ratio evolution
"""

# pylint: disable=invalid-name

import numpy as np

from PySDM.environments import Parcel


def Rv_prim(*, Rl, Nl, Rv, Nv, dNl, Rr, K, a):
    """eq. (2) in [Rozanski and Sonntag 1982](https://doi.org/10.3402/tellusa.v34i2.10795)"""
    return (Rl * Nl + Rv * Nv + dNl * Rr * K) / ((Nl + dNl * K) * a + Nv)


class MultiBox(Parcel):
    """iterative parcel model in which each new iteration operates with ambient isotopic profile
    resultant from the previous iteration, leading to a stationary state
    """

    def __init__(
        self,
        *,
        isotopes,
        delta_nl,
        rain_isotope_ratios,
        nt,
        autoconversion_mixrat_threshold,
        isotope_exchange_factor,
        **kwargs,
    ):
        super().__init__(
            variables=[
                f"{ratio}_{isotope}" for ratio in ("Rv", "Rr") for isotope in isotopes
            ],
            **kwargs,
        )
        self.isotopes = isotopes
        self.delta_nl = delta_nl
        self.rain_isotope_ratios = rain_isotope_ratios
        self.nt = nt
        self.autoconversion_mixrat_threshold = autoconversion_mixrat_threshold
        self.isotope_exchange_factor = isotope_exchange_factor

    def advance_parcel_vars(self):
        """explicit Euler integration of isotope-ratio time derivative"""
        assert self.delta_liquid_water_mixing_ratio >= 0
        self._recalculate_temperature_pressure_relative_humidity(self._tmp)

        alpha_old = {}
        dRv__dt = {}
        for isotope in self.isotopes:
            alpha_fun = getattr(
                self.particulator.formulae.isotope_equilibrium_fractionation_factors,
                f"alpha_l_{isotope}",
            )
            alpha_old[isotope] = alpha_fun(self["T"][0])
            alpha_new = alpha_fun(self._tmp["T"][0])

            dRv__dt[isotope] = self[f"Rv_{isotope}"][
                0
            ] * self.particulator.formulae.isotope_ratio_evolution.d_Rv_over_Rv(
                alpha=alpha_old[isotope],
                d_alpha=(alpha_new - alpha_old[isotope]) / self.dt,
                n_vapour=self["water_vapour_mixing_ratio"][0],
                d_n_vapour=-self.delta_liquid_water_mixing_ratio / self.dt,
                n_liquid=self.autoconversion_mixrat_threshold,  # TODO #1207
            )

        super().advance_parcel_vars()

        for isotope in self.isotopes:
            self.particulator.backend.explicit_euler(
                self._tmp[f"Rv_{isotope}"], self.particulator.dt, dRv__dt[isotope]
            )
            level = self.particulator.n_steps
            if self.delta_nl is not None:
                self._tmp[f"Rv_{isotope}"][:] = Rv_prim(
                    Rl=alpha_old[isotope] * self._tmp[f"Rv_{isotope}"][0],
                    Nl=self.autoconversion_mixrat_threshold,
                    Rv=self._tmp[f"Rv_{isotope}"][0],
                    Nv=self["water_vapour_mixing_ratio"][0],
                    dNl=np.sum(self.delta_nl[level:]),
                    Rr=self.rain_isotope_ratios[isotope][
                        min(
                            level + 2, self.nt  # TODO #1207: this warrants a comment...
                        )
                    ],
                    K=self.isotope_exchange_factor,
                    a=alpha_old[isotope],
                )
            self._tmp[f"Rr_{isotope}"][:] = (
                alpha_old[isotope] * self._tmp[f"Rv_{isotope}"][0]
            )
