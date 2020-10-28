"""
Created at 18.11.2019
"""

import numpy as np
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.settings import Settings
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.simulation import Simulation
from PySDM.physics.constants import si
from PySDM.physics import formulae as phys
from matplotlib import pyplot
import pytest


@pytest.mark.skip  # TODO: sometimes fails... (https://travis-ci.org/atmos-cloud-sim-uj/PySDM/jobs/651243742#L454)
def test_initialisation(plot=False):
    # TODO: seed as a part of settings?
    settings = Settings()
    settings.n_steps = -1
    settings.grid = (10, 5)
    settings.n_sd_per_gridbox = 2000

    simulation = Simulation(settings, None)

    n_bins = 32
    n_levels = settings.grid[1]
    n_cell = np.prod(np.array(settings.grid))
    n_moments = 1

    v_bins = np.logspace(
        (np.log10(phys.volume(radius=settings.r_min))),
        (np.log10(phys.volume(radius=10*settings.r_max))),
        num=n_bins,
        endpoint=True
    )
    r_bins = phys.radius(volume=v_bins)

    histogram_dry = np.empty((len(r_bins) - 1, n_levels))
    histogram_wet = np.empty_like(histogram_dry)

    moment_0 = settings.backend.Storage.empty(n_cell, dtype=int)
    moments = settings.backend.Storage.empty((n_moments, n_cell), dtype=float)
    tmp = np.empty(n_cell)
    simulation.reinit()

    # Act (moments)
    simulation.run()
    particles = simulation.core
    environment = simulation.core.environment
    rhod = environment["rhod"].to_ndarray().reshape(settings.grid).mean(axis=0)

    for i in range(len(histogram_dry)):
        particles.particles.moments(
            moment_0, moments, specs={}, attr_name='dry volume', attr_range=(v_bins[i], v_bins[i + 1]))
        moment_0.download(tmp)
        histogram_dry[i, :] = tmp.reshape(settings.grid).sum(axis=0) / (particles.mesh.dv * settings.grid[0])

        particles.particles.moments(
            moment_0, moments, specs={}, attr_name='volume', attr_range=(v_bins[i], v_bins[i + 1]))
        moment_0.download(tmp)
        histogram_wet[i, :] = tmp.reshape(settings.grid).sum(axis=0) / (particles.mesh.dv * settings.grid[0])

    # Plot
    if plot:
        for level in range(0, n_levels):
            color = str(.5 * (2 + (level / (n_levels - 1))))
            pyplot.step(
                r_bins[:-1] * si.metres / si.micrometres,
                histogram_dry[:, level] / si.metre ** 3 * si.centimetre ** 3,
                where='post',
                color=color,
                label="level " + str(level)
            )
            pyplot.step(
                r_bins[:-1] * si.metres / si.micrometres,
                histogram_wet[:, level] / si.metre ** 3 * si.centimetre ** 3,
                where='post',
                color=color,
                linestyle='--'
            )
        pyplot.grid()
        pyplot.xscale('log')
        pyplot.xlabel('particle radius [µm]')
        pyplot.ylabel('concentration per bin [cm^{-3}]')
        pyplot.legend()
        pyplot.show()

    # Assert - location of maximum
    for level in range(n_levels):
        real_max = settings.spectrum_per_mass_of_dry_air.distribution_params[2]
        idx_max_dry = np.argmax(histogram_dry[:, level])
        idx_max_wet = np.argmax(histogram_wet[:, level])
        assert r_bins[idx_max_dry] < real_max < r_bins[idx_max_dry+1]
        assert idx_max_dry < idx_max_wet

    # Assert - total number
    for level in reversed(range(n_levels)):
        mass_conc_dry = np.sum(histogram_dry[:, level]) / rhod[level]
        mass_conc_wet = np.sum(histogram_wet[:, level]) / rhod[level]
        mass_conc_STP = settings.spectrum_per_mass_of_dry_air.norm_factor
        assert .5 * mass_conc_STP < mass_conc_dry < 1.5 * mass_conc_STP
        np.testing.assert_approx_equal(mass_conc_dry, mass_conc_wet)

    # Assert - decreasing number density
    total_above = 0
    for level in reversed(range(n_levels)):
        total_below = np.sum(histogram_dry[:, level])
        assert total_below > total_above
        total_above = total_below
