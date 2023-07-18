import numpy as np
from PySDM_examples.Shipway_and_Hill_2012.settings import Settings as Settings_Shipway

from PySDM.physics import si


class Settings(Settings_Shipway):
    def __init__(
        self,
        n_sd_per_gridbox: int = 256,
        rho_times_w_1: float = 2 * si.m / si.s * si.kg / si.m**3,
        precip: bool = True,
        evaluate_relaxed_velocity=True,
        tau=1 * si.second,
        times_to_save=None,
    ):
        super().__init__(
            n_sd_per_gridbox=n_sd_per_gridbox,
            rho_times_w_1=rho_times_w_1,
            precip=precip,
            kappa=0.9,
            dt=5 * si.s,
            dz=50 * si.m,
            p0=990 * si.hPa,
        )
        self.evaluate_relaxed_velocity = evaluate_relaxed_velocity
        self.tau = tau
        self.save_spec_and_attr_times = (
            np.array([]) if times_to_save is None else times_to_save
        )
