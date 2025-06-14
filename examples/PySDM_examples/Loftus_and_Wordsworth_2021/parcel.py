from PySDM.environments.parcel import Parcel


class AlienParcel(Parcel):
    def __init__(
        self,
        dt,
        mass_of_dry_air: float,
        pcloud: float,
        initial_water_vapour_mixing_ratio: float,
        Tcloud: float,
        w: float = 0,
        zcloud: float = 0,
        mixed_phase=False,
    ):
        super().__init__(
            dt=dt,
            mass_of_dry_air=mass_of_dry_air,
            p0=pcloud,
            initial_water_vapour_mixing_ratio=initial_water_vapour_mixing_ratio,
            T0=Tcloud,
            w=w,
            z0=zcloud,
            mixed_phase=mixed_phase,
            variables=None,
        )

    def advance_parcel_vars(self):
        """
        Compute new values of displacement, dry-air density and volume,
        and write them to self._tmp and self.mesh.dv
        """
        dt = self.particulator.dt
        formulae = self.particulator.formulae
        T = self["T"][0]
        p = self["p"][0]

        dz_dt = -self.particulator.attributes["terminal velocity"].to_ndarray()[0]
        water_vapour_mixing_ratio = (
            self["water_vapour_mixing_ratio"][0]
            - self.delta_liquid_water_mixing_ratio / 2
        )

        drho_dz = formulae.hydrostatics.drho_dz(
            p=p,
            T=T,
            water_vapour_mixing_ratio=water_vapour_mixing_ratio,
            lv=formulae.latent_heat_vapourisation.lv(T),
            d_liquid_water_mixing_ratio__dz=(
                self.delta_liquid_water_mixing_ratio / dz_dt / dt
            ),
        )
        drhod_dz = drho_dz

        self.particulator.backend.explicit_euler(self._tmp["z"], dt, dz_dt)
        self.particulator.backend.explicit_euler(
            self._tmp["rhod"], dt, dz_dt * drhod_dz
        )

        self.mesh.dv = formulae.trivia.volume_of_density_mass(
            (self._tmp["rhod"][0] + self["rhod"][0]) / 2, self.mass_of_dry_air
        )
