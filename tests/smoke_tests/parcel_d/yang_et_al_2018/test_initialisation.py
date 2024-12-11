# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import matplotlib.pyplot as plt
import numpy as np
from PySDM_examples.Yang_et_al_2018 import Settings, Simulation

from PySDM.physics.constants import si


def test_dry_spectrum_x():
    settings = Settings()
    simulation = Simulation(settings)
    dry_volume = simulation.particulator.attributes["dry volume"].to_ndarray()
    dry_radii = simulation.formulae.trivia.radius(volume=dry_volume) / si.nanometre

    dry_radii = dry_radii[::-1]
    assert round(dry_radii[1 - 1], 0) == 503
    assert round(dry_radii[10 - 1], 0) == 355
    assert round(dry_radii[50 - 1], 1) == 75.3
    assert round(dry_radii[100 - 1], 1) == 10.8


def test_dry_spectrum_y(plot=False):
    settings = Settings()
    simulation = Simulation(settings)
    dry_volume = simulation.particulator.attributes["dry volume"].to_ndarray()
    dry_radii = simulation.formulae.trivia.radius(volume=dry_volume) / si.nanometre
    nd = simulation.particulator.attributes["multiplicity"].to_ndarray()

    dr = dry_radii[1:] - dry_radii[0:-1]
    env = simulation.particulator.environment
    dn_dr = nd[0:-1] / env.mass_of_dry_air * env["rhod"] / dr
    dn_dr /= 1 / si.centimetre**3

    if plot:
        plt.figure(figsize=(5, 5))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(1e1, 1e3)
        plt.ylim(1e-9, 1e3)
        plt.yticks(10.0 ** np.arange(-8, 3, step=2))
        plt.plot(dry_radii[0:-1], dn_dr)
        plt.show()

    # from fig. 1b
    assert 1e-3 < dn_dr[0] < 1e-2
    assert 1e1 < max(dn_dr) < 1e2
    assert 0 < dn_dr[-1] < 1e-9


def test_wet_vs_dry_spectrum(plot=False):
    # Arrange
    settings = Settings()

    # Act
    simulation = Simulation(settings)
    wet_volume = simulation.particulator.attributes["volume"].to_ndarray()
    r_wet = simulation.formulae.trivia.radius(volume=wet_volume) / si.nanometre
    n = simulation.particulator.attributes["multiplicity"].to_ndarray()

    dry_volume = simulation.particulator.attributes["dry volume"].to_ndarray()
    dry_radii = simulation.formulae.trivia.radius(volume=dry_volume) / si.nanometre

    # Plot
    if plot:
        plt.plot(r_wet, n)
        plt.plot(dry_radii, n)
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    # Assert
    assert (dry_radii < r_wet).all()


def test_relative_humidity():
    # Arrange
    settings = Settings()

    # Act
    simulation = Simulation(settings)

    # Assert
    assert round(simulation.particulator.environment["RH"][0], 3) == 0.856
