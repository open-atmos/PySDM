import numpy as np
from PySDM_examples.Lowe_et_al_2019.constants_def import LOWE_CONSTS
from pystrict import strict

from PySDM import Formulae
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.initialisation.sampling import spectral_sampling as spec_sampling
from PySDM.physics import si


@strict
class Settings:
    def __init__(
        self,
        dz: float,
        n_sd_per_mode: int,
        aerosol: DryAerosolMixture,
        model: str,
        spectral_sampling: type(spec_sampling.SpectralSampling),
        w: float = 0.32 * si.m / si.s,
    ):
        assert model in ("Constant", "CompressedFilmOvadnevaite")
        self.model = model
        self.n_sd_per_mode = n_sd_per_mode
        self.formulae = Formulae(
            surface_tension=model,
            constants=LOWE_CONSTS,
            diffusion_kinetics="LoweEtAl2019",
            diffusion_thermics="LoweEtAl2019",
            latent_heat="Lowe2019",
            saturation_vapour_pressure="Lowe1977",
            optical_albedo="Bohren1987",
            optical_depth="Stephens1978",
        )
        const = self.formulae.constants
        self.aerosol = aerosol
        self.spectral_sampling = spectral_sampling

        max_altitude = 200 * si.m
        self.w = w
        self.t_max = max_altitude / self.w
        self.dt = dz / self.w
        self.output_interval = 1 * self.dt

        self.p0 = 980 * si.mbar
        self.T0 = 280 * si.K
        pv0 = 0.999 * self.formulae.saturation_vapour_pressure.pvs_Celsius(
            self.T0 - const.T0
        )
        self.initial_water_vapour_mixing_ratio = const.eps * pv0 / (self.p0 - pv0)

        self.cloud_radius_range = (0.5 * si.micrometre, np.inf)

        self.mass_of_dry_air = 44

        self.wet_radius_bins_edges = np.logspace(
            np.log10(4 * si.um), np.log10(12 * si.um), 128 + 1, endpoint=True
        )

    @property
    def rho0(self):
        const = self.formulae.constants
        rhod0 = (
            self.formulae.trivia.p_d(self.p0, self.initial_water_vapour_mixing_ratio)
            / self.T0
            / const.Rd
        )
        return rhod0 * (1 + self.initial_water_vapour_mixing_ratio)

    @property
    def nt(self) -> int:
        nt = self.t_max / self.dt
        nt_int = round(nt)
        np.testing.assert_almost_equal(nt, nt_int)
        return nt_int

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.dt)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.nt + 1, self.steps_per_output_interval)
