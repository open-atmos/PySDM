"""
Zero-dimensional expansion chamber framework
"""

import numpy as np

from PySDM.environments.impl.moist import Moist
from PySDM.impl.mesh import Mesh
from PySDM.initialisation.equilibrate_wet_radii import (
    default_rtol,
    equilibrate_wet_radii,
)
from PySDM.environments.impl import register_environment


@register_environment()
class ExpansionChamber(Moist):
    def __init__(
        self,
        *,
        dt,
        volume: float,
        p0: float,
        T0: float,
        initial_water_vapour_mixing_ratio: float,
        pf: float,
        tf: float,
        variables,
        mixed_phase=False,
    ):
        super().__init__(dt, Mesh.mesh_0d(), variables, mixed_phase=mixed_phase)

        self.volume = volume
        self.p0 = p0
        self.T0 = T0
        self.initial_water_vapour_mixing_ratio = initial_water_vapour_mixing_ratio
        self.pf = pf
        self.tf = tf

        self.formulae = None
        self.delta_liquid_water_mixing_ratio = np.nan
        self.params = None

    def register(self, builder):
        self.formulae = builder.particulator.formulae
        pd0 = self.formulae.trivia.p_d(self.p0, self.initial_water_vapour_mixing_ratio)
        rhod0 = self.formulae.state_variable_triplet.rhod_of_pd_T(pd0, self.T0)
        self.mesh.dv = self.volume

        Moist.register(self, builder)

        params = (self.p0, self.T0, self.initial_water_vapour_mixing_ratio, 0)
        self["p"][:] = params[0]
        self["T"][:] = params[1]
        self["water_vapour_mixing_ratio"][:] = params[2]
        self["t"][:] = params[3]

        self._tmp["water_vapour_mixing_ratio"][:] = params[0]
        self.sync_chamber_vars()
        Moist.sync(self)
        self.notify()

    def init_attributes(
        self,
        *,
        n_in_dv: [float, np.ndarray],
        kappa: float,
        r_dry: [float, np.ndarray],
        rtol=default_rtol,
        include_dry_volume_in_attribute: bool = True,
    ):
        if not isinstance(n_in_dv, np.ndarray):
            r_dry = np.array([r_dry])
            n_in_dv = np.array([n_in_dv])

        attributes = {}
        dry_volume = self.formulae.trivia.volume(radius=r_dry)
        attributes["kappa times dry volume"] = dry_volume * kappa
        attributes["multiplicity"] = n_in_dv
        r_wet = equilibrate_wet_radii(
            r_dry=r_dry,
            environment=self,
            kappa_times_dry_volume=attributes["kappa times dry volume"],
            rtol=rtol,
        )
        attributes["volume"] = self.formulae.trivia.volume(radius=r_wet)
        if include_dry_volume_in_attribute:
            attributes["dry volume"] = dry_volume
        return attributes

    def formulae_drho_dp(const, p, T, water_vapour_mixing_ratio, lv, dql_dp):
        R_q = const.Rv / (1 / water_vapour_mixing_ratio + 1) + const.Rd / (
            1 + water_vapour_mixing_ratio
        )
        cp_q = const.c_pv / (1 / water_vapour_mixing_ratio + 1) + const.c_pd / (
            1 + water_vapour_mixing_ratio
        )
        return -1 / R_q / T - p / R_q / T**2 * lv / cp_q * dql_dp

    def advance_chamber_vars(self):
        dt = self.particulator.dt
        T = self["T"][0]
        p = self["p"][0]
        t = self["t"][0]

        dp_dt = (self.pf - self.p0) / (self.tf) * (self.dt / 2)  # mid-point
        water_vapour_mixing_ratio = (
            self["water_vapour_mixing_ratio"][0]
            - self.delta_liquid_water_mixing_ratio / 2
        )

        drho_dp = self.formulae_drho_dp(
            p=p,
            T=T,
            water_vapour_mixing_ratio=water_vapour_mixing_ratio,
            lv=self.formulae.latent_heat.lv(T),
            dql_dp=self.delta_liquid_water_mixing_ratio / dt,
        )
        drhod_dp = drho_dp

        self.particulator.backend.explicit_euler(self._tmp["t"], dt, 1)
        self.particulator.backend.explicit_euler(self._tmp["p"], dt, dp_dt)
        self.particulator.backend.explicit_euler(
            self._tmp["rhod"], dt, drhod_dp * dp_dt
        )

    def sync_chamber_vars(self):
        self.delta_liquid_water_mixing_ratio = (
            self._tmp["water_vapour_mixing_ratio"][0]
            - self["water_vapour_mixing_ratio"][0]
        )
        for var in self.variables:
            self._tmp[var][:] = self[var][:]

    def sync(self):
        self.sync_chamber_vars()
        self.advance_chamber_vars()
        super().sync()
