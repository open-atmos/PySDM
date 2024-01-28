import numpy as np
from chempy import Substance
from pystrict import strict

from PySDM import Formulae
from PySDM.dynamics.impl.chemistry_utils import AQUEOUS_COMPOUNDS
from PySDM.initialisation import spectra
from PySDM.initialisation.sampling import spectral_sampling as spec_sampling
from PySDM.physics import si
from PySDM.physics.constants import PPB, PPM, T0


@strict
class Settings:
    def __init__(
        self,
        dt: float,
        n_sd: int,
        n_substep: int,
        spectral_sampling: spec_sampling.SpectralSampling = spec_sampling.Logarithmic,
    ):
        self.formulae = Formulae(
            saturation_vapour_pressure="AugustRocheMagnus",
            constants={"g_std": 10 * si.m / si.s**2},
        )
        const = self.formulae.constants
        self.DRY_RHO = 1800 * si.kg / (si.m**3)
        self.dry_molar_mass = Substance.from_formula("NH4HSO4").mass * si.gram / si.mole

        self.system_type = "closed"

        self.t_max = (2400 + 196) * si.s
        self.output_interval = 10 * si.s
        self.dt = dt

        self.w = 0.5 * si.m / si.s

        self.n_sd = n_sd
        self.n_substep = n_substep

        self.p0 = 950 * si.mbar
        self.T0 = 285.2 * si.K
        pv0 = 0.95 * self.formulae.saturation_vapour_pressure.pvs_Celsius(self.T0 - T0)
        self.initial_water_vapour_mixing_ratio = const.eps * pv0 / (self.p0 - pv0)
        self.kappa = 0.61

        self.cloud_radius_range = (0.5 * si.micrometre, 25 * si.micrometre)

        self.mass_of_dry_air = 44

        # note: rho is not specified in the paper
        rho0 = 1

        self.r_dry, self.n_in_dv = spectral_sampling(
            spectrum=spectra.Lognormal(
                norm_factor=566 / si.cm**3 / rho0 * self.mass_of_dry_air,
                m_mode=0.08 * si.um / 2,
                s_geom=2,
            )
        ).sample(n_sd)

        self.ENVIRONMENT_MOLE_FRACTIONS = {
            "SO2": 0.2 * PPB,
            "O3": 50 * PPB,
            "H2O2": 0.5 * PPB,
            "CO2": 360 * PPM,
            "HNO3": 0.1 * PPB,
            "NH3": 0.1 * PPB,
        }

        self.starting_amounts = {
            "moles_"
            + k: (
                self.formulae.trivia.volume(self.r_dry)
                * self.DRY_RHO
                / self.dry_molar_mass
                if k in ("N_mIII", "S_VI")
                else np.zeros(self.n_sd)
            )
            for k in AQUEOUS_COMPOUNDS
        }

        self.dry_radius_bins_edges = (
            np.logspace(np.log10(0.01 * si.um), np.log10(1 * si.um), 51, endpoint=True)
            / 2
        )

    @property
    def nt(self):
        nt = self.t_max / self.dt
        assert nt == int(nt)
        return int(nt)

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.dt)
