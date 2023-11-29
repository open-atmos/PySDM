import numba
import numpy
import numpy as np  # pylint: disable=reimported
import PyMPDATA
import scipy

import PySDM
from PySDM import Formulae
from PySDM.dynamics import collisions, condensation, displacement
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.collision_kernels import Geometric
from PySDM.initialisation import spectra
from PySDM.physics import si


class Common:
    def __init__(self, formulae: Formulae):
        self.formulae = formulae
        const = formulae.constants

        self.condensation_rtol_x = condensation.DEFAULTS.rtol_x
        self.condensation_rtol_thd = condensation.DEFAULTS.rtol_thd
        self.condensation_adaptive = True
        self.condensation_substeps = -1
        self.condensation_dt_cond_range = condensation.DEFAULTS.cond_range
        self.condensation_schedule = condensation.DEFAULTS.schedule

        self.coalescence_adaptive = True
        self.coalescence_dt_coal_range = collisions.collision.DEFAULTS.dt_coal_range
        self.coalescence_optimized_random = True
        self.coalescence_substeps = 1
        self.kernel = Geometric(collection_efficiency=1)
        self.coalescence_efficiency = ConstEc(Ec=1.0)
        self.breakup_efficiency = ConstEb(Eb=1.0)
        self.breakup_fragmentation = AlwaysN(n=2)

        self.freezing_singular = True
        self.freezing_thaw = False
        self.freezing_inp_spec = None

        self.displacement_adaptive = displacement.DEFAULTS.adaptive
        self.displacement_rtol = displacement.DEFAULTS.rtol
        self.freezing_inp_frac = 1

        self.n_sd_per_gridbox = 20

        self.aerosol_radius_threshold = 0.5 * si.micrometre
        self.drizzle_radius_threshold = 25 * si.micrometre

        self.r_bins_edges = np.logspace(
            np.log10(0.001 * si.micrometre),
            np.log10(100 * si.micrometre),
            64,
            endpoint=True,
        )
        self.T_bins_edges = np.linspace(const.T0 - 40, const.T0 - 20, 64, endpoint=True)

        # TODO #599
        n_bins_per_phase = 25
        solid_phase_radii = (
            np.linspace(-n_bins_per_phase, -1, n_bins_per_phase + 1) * si.um
        )
        liquid_phase_radii = (
            np.linspace(0, n_bins_per_phase, n_bins_per_phase + 1) * si.um
        )
        self.terminal_velocity_radius_bin_edges = np.concatenate(
            [solid_phase_radii, liquid_phase_radii]
        )

        self.output_interval = 1 * si.minute
        self.spin_up_time = 0

        self.mode_1 = spectra.Lognormal(
            norm_factor=60 / si.centimetre**3 / const.rho_STP,
            m_mode=0.04 * si.micrometre,
            s_geom=1.4,
        )
        self.mode_2 = spectra.Lognormal(
            norm_factor=40 / si.centimetre**3 / const.rho_STP,
            m_mode=0.15 * si.micrometre,
            s_geom=1.6,
        )
        self.spectrum_per_mass_of_dry_air = spectra.Sum((self.mode_1, self.mode_2))
        self.kappa = 1  # TODO #441!

        self.processes = {
            "particle advection": True,
            "fluid advection": True,
            "coalescence": True,
            "condensation": True,
            "sedimentation": True,
            "breakup": False,
            "freezing": False,
        }

        self.mpdata_iters = 2
        self.mpdata_iga = True
        self.mpdata_fct = True
        self.mpdata_tot = True

        key_packages = [PySDM, PyMPDATA, numba, numpy, scipy]
        try:
            import ThrustRTC  # pylint: disable=import-outside-toplevel

            key_packages.append(ThrustRTC)
        except:  # pylint: disable=bare-except
            pass
        self.versions = {}
        for pkg in key_packages:
            try:
                self.versions[pkg.__name__] = pkg.__version__
            except AttributeError:
                pass
        self.versions = str(self.versions)

        self.dt = None
        self.simulation_time = None
        self.grid = None
        self.p0 = None
        self.initial_water_vapour_mixing_ratio = None
        self.th_std0 = None
        self.size = None

    @property
    def n_steps(self) -> int:
        return int(self.simulation_time / self.dt)  # TODO #413

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.dt)

    @property
    def n_spin_up(self) -> int:
        return int(self.spin_up_time / self.dt)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.n_steps + 1, self.steps_per_output_interval)

    @property
    def n_sd(self):
        return self.grid[0] * self.grid[1] * self.n_sd_per_gridbox

    @property
    def initial_vapour_mixing_ratio_profile(self):
        return np.full(self.grid[-1], self.initial_water_vapour_mixing_ratio)

    @property
    def initial_dry_potential_temperature_profile(self):
        return np.full(
            self.grid[-1],
            self.formulae.state_variable_triplet.th_dry(
                self.th_std0, self.initial_water_vapour_mixing_ratio
            ),
        )
