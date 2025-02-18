"""
Zero-dimensional expansion chamber framework
"""

from typing import Optional, List

from PySDM.environments.impl.moist_lagrangian import MoistLagrangian
from PySDM.impl.mesh import Mesh
from PySDM.environments.impl import register_environment


@register_environment()
class ExpansionChamber(MoistLagrangian):
    def __init__(
        self,
        *,
        dt,
        volume: float,
        initial_pressure: float,
        initial_temperature: float,
        initial_relative_humidity: float,
        delta_pressure: float,
        delta_time: float,
        variables: Optional[List[str]] = None,
        mixed_phase=False,
    ):
        variables = (variables or []) + ["rhod"]

        super().__init__(dt, Mesh.mesh_0d(), variables, mixed_phase=mixed_phase)

        self.dv = volume
        self.initial_pressure = initial_pressure
        self.initial_temperature = initial_temperature
        self.initial_relative_humidity = initial_relative_humidity
        self.delta_time = delta_time
        self.dp_dt = delta_pressure / delta_time

    def register(self, builder):
        self.mesh.dv = self.dv

        super().register(builder)

        formulae = self.particulator.formulae
        pv0 = (
            self.initial_relative_humidity
            * formulae.saturation_vapour_pressure.pvs_water(self.initial_temperature)
        )
        th_std = formulae.trivia.th_std(
            p=self.initial_pressure, T=self.initial_temperature
        )
        initial_water_vapour_mixing_ratio = (
            formulae.constants.eps * pv0 / (self.initial_pressure - pv0)
        )

        self["rhod"][:] = formulae.state_variable_triplet.rho_d(
            p=self.initial_pressure,
            water_vapour_mixing_ratio=initial_water_vapour_mixing_ratio,
            theta_std=th_std,
        )
        self["thd"][:] = formulae.state_variable_triplet.th_dry(
            th_std, initial_water_vapour_mixing_ratio
        )
        self["water_vapour_mixing_ratio"][:] = initial_water_vapour_mixing_ratio

        self.post_register()

    def advance_moist_vars(self):
        """compute new values of dry density and thd, and write them to self._tmp and self["thd"]
        assuming water-vapour mixing ratio is not altered by the expansion"""
        dt = self.particulator.dt
        t_new = (self.particulator.n_steps + 1) * dt
        if t_new > self.delta_time:
            return

        formulae = self.particulator.formulae
        p_new = self["p"][0] + self.dp_dt * dt
        vapour_mixing_ratio = self["water_vapour_mixing_ratio"][0]
        gg = (
            1 - formulae.adiabatic_exponent.gamma(vapour_mixing_ratio)
        ) / formulae.adiabatic_exponent.gamma(vapour_mixing_ratio)
        T_new = self.initial_temperature * (self.initial_pressure / p_new) ** gg
        wvmr_new = self._tmp["water_vapour_mixing_ratio"][
            0
        ]  # TODO #1492 - should _tmp or self[] be used?
        pv_new = wvmr_new * p_new / (wvmr_new + formulae.constants.eps)
        pd_new = p_new - pv_new

        self._tmp["rhod"][:] = pd_new / T_new / formulae.constants.Rd
        self["thd"][:] = formulae.trivia.th_std(p=pd_new, T=T_new)

    def sync_moist_vars(self):
        for var in self.variables:
            self._tmp[var][:] = self[var][:]
