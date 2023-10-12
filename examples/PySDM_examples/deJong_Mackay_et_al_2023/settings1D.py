from typing import Iterable

import numpy as np
from PySDM_examples.Shipway_and_Hill_2012 import Settings as SettingsSH

from PySDM import Formulae
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import Gaussian, Straub2010Nf
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc, Straub2010Ec
from PySDM.physics import si


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
            "breakup",
            "stochastic_breakup",
            "z_max",
            "t_max",
            "cloud_water_radius_range",
            "rain_water_radius_range",
            "r_bins_edges_dry",
            "r_bins_edges",
            "save_spec_at",
        )

    def __init__(
        self,
        *,
        n_sd_per_gridbox: int,
        p0: float = 1007 * si.hPa,  # as used in Olesik et al. 2022 (GMD)
        kappa: float = 1,
        rho_times_w_1: float = 2 * si.m / si.s * si.kg / si.m**3,
        particles_per_volume_STP: int = 50 / si.cm**3,
        dt: float = 1 * si.s,
        dz: float = 25 * si.m,
        z_max: float = 3000 * si.m,
        t_max: float = 3600 * si.s,
        precip: bool = True,
        breakup: bool = False,
        stochastic_breakup: bool = False,
        warn_breakup_overflow: bool = False,
        output_every_n_steps: int = 1,
        save_spec_at=()
    ):
        if stochastic_breakup:
            self.coalescence_efficiency = Straub2010Ec()
            self.fragmentation_function = Straub2010Nf(vmin=1 * si.um**3)
        else:
            self.coalescence_efficiency = ConstEc(Ec=0.95)
            frag_scale_r = 30 * si.um
            frag_scale_v = frag_scale_r**3 * 4 / 3 * np.pi
            self.fragmentation_function = Gaussian(
                mu=frag_scale_v,
                sigma=frag_scale_v / 2,
                vmin=(1 * si.um) ** 3,
                nfmax=20,
            )

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
            formulae=Formulae(
                fragmentation_function=self.fragmentation_function.__class__.__name__
            ),
            save_spec_and_attr_times=save_spec_at,
        )
        self.breakup = breakup
        self.stochastic_breakup = stochastic_breakup

        self.breakup_efficiency = ConstEb(Eb=1.0)

        self.warn_breakup_overflow = warn_breakup_overflow
        self.output_steps = range(0, self.nt, output_every_n_steps)
