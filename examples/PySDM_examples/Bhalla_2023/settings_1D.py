from PySDM.physics import si

from PySDM_examples.Shipway_and_Hill_2012.settings import Settings as Settings_Shipway


class Settings(Settings_Shipway):
    def __init__(
        self,
        n_sd_per_gridbox: int = 256,
        rho_times_w_1: float = 2 * si.m / si.s * si.kg / si.m**3,
        precip: bool = True
    ):
        super().__init__(
            n_sd_per_gridbox=n_sd_per_gridbox,
            rho_times_w_1=rho_times_w_1,
            precip=precip,
            kappa=0.9,
            dt=5 * si.s,
            dz=50 * si.m,
            p0=990 * si.hPa
        )
