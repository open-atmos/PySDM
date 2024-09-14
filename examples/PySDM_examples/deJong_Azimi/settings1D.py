from typing import Iterable

from PySDM_examples.Shipway_and_Hill_2012 import Settings as SettingsSH

from PySDM import Formulae
from PySDM.initialisation import spectra
from PySDM.physics import si
from PySDM.dynamics.collisions.collision_kernels import Golovin


class Settings1D(SettingsSH):
    def __dir__(self) -> Iterable[str]:
        return (
            "n_sd_per_gridbox",
            "p0",
            "kappa",
            "rho_times_w_1",
            "particles_per_volume_STP",
            "dt",
            "dz",
            "precip",
            "z_max",
            "t_max",
            "cloud_water_radius_range",
            "rain_water_radius_range",
            "r_bins_edges_dry",
            "r_bins_edges",
        )

    def __init__(
        self,
        *,
        n_sd_per_gridbox: int,
        p0: float = 1007 * si.hPa,  # as used in Olesik et al. 2022 (GMD)
        kappa: float = 1,
        rho_times_w_1: float = 2 * si.m / si.s * si.kg / si.m**3,
        particles_per_volume_STP: int = 100 / si.cm**3,
        dt: float = 1 * si.s,
        dz: float = 25 * si.m,
        z_max: float = 3000 * si.m,
        t_max: float = 60 * si.minutes,
        precip: bool = True,
        formulae: Formulae = None,
        save_spec_and_attr_times=(),
        z_part=(0.5, 0.75)
    ):
        super().__init__(
            n_sd_per_gridbox=n_sd_per_gridbox,
            p0=p0,
            kappa=kappa,
            rho_times_w_1=rho_times_w_1,
            particles_per_volume_STP=particles_per_volume_STP,
            dt=dt,
            dz=dz,
            z_max=z_max,
            t_max=t_max,
            precip=precip,
            formulae=formulae or Formulae(terminal_velocity="PowerSeries"),
            save_spec_and_attr_times=save_spec_and_attr_times,
            enable_condensation=False,
            collision_kernel=Golovin(b=5e3),
        )
        self.z_part = z_part
        z_frac = z_part[1] - z_part[0]
        norm_factor = (
            particles_per_volume_STP / self.formulae.constants.rho_STP * z_frac
        )
        self.wet_radius_spectrum_per_mass_of_dry_air = spectra.Gamma(
            norm_factor=norm_factor,
            k=1.0,
            theta=1e5 * si.um**3,
        )
