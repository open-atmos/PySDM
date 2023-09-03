# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from matplotlib import pyplot
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM_examples.Szumowski_et_al_1998 import Simulation

from PySDM.physics.constants import si


# pylint: disable=too-many-locals
def test_initialisation(backend_class, plot=False):
    settings = Settings()
    settings.simulation_time = -1 * settings.dt
    settings.grid = (10, 5)
    settings.n_sd_per_gridbox = 5000

    simulation = Simulation(settings, None, SpinUp=SpinUp, backend_class=backend_class)

    n_levels = settings.grid[1]
    n_cell = int(np.prod(np.array(settings.grid)))
    n_moments = 1

    r_bins = settings.r_bins_edges

    histogram_dry = np.empty((len(r_bins) - 1, n_levels))
    histogram_wet = np.empty_like(histogram_dry)

    tmp = np.empty(n_cell)
    simulation.reinit()

    # Act (moments)
    simulation.run()
    particulator = simulation.particulator
    environment = simulation.particulator.environment
    rhod = environment["rhod"].to_ndarray().reshape(settings.grid).mean(axis=0)

    v_bins = settings.formulae.trivia.volume(settings.r_bins_edges)

    moment_0 = particulator.backend.Storage.empty(n_cell, dtype=float)
    moments = particulator.backend.Storage.empty((n_moments, n_cell), dtype=float)
    for i in range(len(v_bins) - 1):
        particulator.moments(
            moment_0=moment_0,
            moments=moments,
            specs={"multiplicity": (0,)},
            attr_name="dry volume",
            attr_range=(v_bins[i], v_bins[i + 1]),
        )
        moment_0.download(tmp)
        histogram_dry[i, :] = tmp.reshape(settings.grid).sum(axis=0) / (
            particulator.mesh.dv * settings.grid[0]
        )

        particulator.moments(
            moment_0=moment_0,
            moments=moments,
            specs={"multiplicity": (0,)},
            attr_name="volume",
            attr_range=(v_bins[i], v_bins[i + 1]),
        )
        moment_0.download(tmp)
        histogram_wet[i, :] = tmp.reshape(settings.grid).sum(axis=0) / (
            particulator.mesh.dv * settings.grid[0]
        )

    # Plot
    for level in range(0, n_levels):
        color = str(0.75 * (level / (n_levels - 1)))
        pyplot.step(
            r_bins[:-1] * si.metres / si.micrometres,
            histogram_dry[:, level] / si.metre**3 * si.centimetre**3,
            where="post",
            color=color,
            label="level " + str(level),
        )
        pyplot.step(
            r_bins[:-1] * si.metres / si.micrometres,
            histogram_wet[:, level] / si.metre**3 * si.centimetre**3,
            where="post",
            color=color,
            linestyle="--",
        )
    pyplot.grid()
    pyplot.xscale("log")
    pyplot.xlabel("particle radius [µm]")
    pyplot.ylabel("concentration per bin [cm^{-3}]")
    pyplot.legend()
    if plot:
        pyplot.show()

    # Assert - total number
    for level in reversed(range(n_levels)):
        mass_conc_dry = np.sum(histogram_dry[:, level]) / rhod[level]
        mass_conc_wet = np.sum(histogram_wet[:, level]) / rhod[level]
        mass_conc_STP = settings.spectrum_per_mass_of_dry_air.norm_factor
        assert 0.5 * mass_conc_STP < mass_conc_dry < 1.5 * mass_conc_STP
        np.testing.assert_approx_equal(mass_conc_dry, mass_conc_wet, significant=5)

    # Assert - decreasing number density
    assert np.sum(histogram_dry[:, 0]) > np.sum(histogram_dry[:, -1])
