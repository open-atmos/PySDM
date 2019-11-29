"""
Created at 18.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from examples.ICMW_2012_case_1.setup import Setup
from examples.ICMW_2012_case_1.example import Simulation
from PySDM.simulation.physics.constants import si
from PySDM.utils import Physics

from matplotlib import pyplot


def test_initialisation(plot=False):
    # TODO: seed as a part of setup?
    setup = Setup()
    setup.steps = []
    setup.grid = (10, 5)
    setup.n_sd_per_gridbox = 2000

    simulation = Simulation(setup, None)

    n_bins = 32
    n_levels = setup.grid[1]
    n_cell = np.prod(np.array(setup.grid))
    n_moments = 1

    x_bins = np.logspace(
        (np.log10(Physics.r2x(setup.r_min))),
        (np.log10(Physics.r2x(10*setup.r_max))),
        num=n_bins,
        endpoint=True
    )
    r_bins = Physics.x2r(x_bins)

    histogram_dry = np.empty((len(r_bins) - 1, n_levels))
    histogram_wet = np.empty_like(histogram_dry)

    moment_0 = setup.backend.array(n_cell, dtype=int)
    moments = setup.backend.array((n_moments, n_cell), dtype=float)
    tmp = np.empty(n_cell)

    # Act (moments)
    simulation.run()
    particles = simulation.particles
    environment = simulation.particles.environment
    rhod = setup.backend.to_ndarray(environment["rhod"]).reshape(setup.grid).mean(axis=0)

    for i in range(len(histogram_dry)):
        particles.state.moments(moment_0, moments, specs={}, attr_name='dry volume', attr_range=(x_bins[i], x_bins[i + 1]))
        particles.backend.download(moment_0, tmp)
        histogram_dry[i, :] = tmp.reshape(setup.grid).sum(axis=0) / (particles.mesh.dv * setup.grid[0])

        particles.state.moments(moment_0, moments, specs={}, attr_name='x', attr_range=(x_bins[i], x_bins[i + 1]))
        particles.backend.download(moment_0, tmp)
        histogram_wet[i, :] = tmp.reshape(setup.grid).sum(axis=0) / (particles.mesh.dv * setup.grid[0])

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
        real_max = setup.spectrum_per_mass_of_dry_air.distribution_params[2]
        idx_max_dry = np.argmax(histogram_dry[:, level])
        idx_max_wet = np.argmax(histogram_wet[:, level])
        assert r_bins[idx_max_dry] < real_max < r_bins[idx_max_dry+1]
        assert idx_max_dry < idx_max_wet

    # Assert - total number
    for level in reversed(range(n_levels)):
        mass_conc_dry = np.sum(histogram_dry[:, level]) / rhod[level]
        mass_conc_wet = np.sum(histogram_wet[:, level]) / rhod[level]
        mass_conc_STP = setup.spectrum_per_mass_of_dry_air.norm_factor
        assert .5 * mass_conc_STP < mass_conc_dry < 1.5 * mass_conc_STP
        np.testing.assert_approx_equal(mass_conc_dry, mass_conc_wet)

    # Assert - decreasing number density
    total_above = 0
    for level in reversed(range(n_levels)):
        total_below = np.sum(histogram_dry[:, level])
        assert total_below > total_above
        total_above = total_below

