"""
Created at 25.11.2019

@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM.simulation.initialisation.spectra import Lognormal
from PySDM.backends.default import Default
from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics.constants import si


class Setup:
    backend = Default

    spectrum_per_mass_of_dry_air = Lognormal(
      norm_factor=1000 / si.milligram, #TODO: was np.sum(n) ???
      m_mode=50 * si.nanometre,
      s_geom=1.4
    )

    n_sd = 100
    T0 = 284.3 * si.kelvin
    q0 = 7.6 * si.grams / si.kilogram
    p0 = 938.5 * si.hectopascals
    # density of ai
    rho = p0 / phys.R(q0) / T0

    # initial dry radius discretisation range
    r_min = 10.633 * si.nanometre
    r_max = 513.06 * si.nanometre


