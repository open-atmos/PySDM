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

    def _compute_dz_dt(self, dt):
        return -self.particulator.attributes["terminal velocity"].to_ndarray()[0]
