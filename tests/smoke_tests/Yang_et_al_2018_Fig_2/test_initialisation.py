from examples.Yang_et_al_2018_Fig_2.example import Simulation
from examples.Yang_et_al_2018_Fig_2.setup import Setup
from PySDM.simulation.physics.constants import si
from PySDM.utils import Physics
import matplotlib.pyplot as plt
import numpy as np


def test_dry_spectrum_x():
    setup = Setup()
    simulation = Simulation(setup)
    dry_volume = simulation.particles.state.get_backend_storage('dry volume')
    dry_volume = simulation.particles.backend.to_ndarray(dry_volume)
    rd = Physics.x2r(dry_volume) / si.nanometre

    rd = rd[::-1]
    assert round(rd[  1-1], 0) == 503
    assert round(rd[ 10-1], 0) == 355
    assert round(rd[ 50-1], 1) == 75.3
    assert round(rd[100-1], 1) == 10.8


def test_dry_spectrum_y():
    setup = Setup()
    simulation = Simulation(setup)
    dry_volume = simulation.particles.state.get_backend_storage('dry volume')
    dry_volume = simulation.particles.backend.to_ndarray(dry_volume)
    rd = Physics.x2r(dry_volume) / si.nanometre
    nd = simulation.particles.backend.to_ndarray(simulation.particles.state.n)

    dr = (rd[1:] - rd[0:-1]) / si.nanometre
    env = simulation.particles.environment
    dn_dr = (nd[0:-1] / env.m_d * env["rhod"] / dr)
    dn_dr /= (1/si.centimetre**3)

    plt.figure(figsize=(5, 5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e1, 1e3)
    plt.ylim(1e-9, 1e3)
    plt.yticks(10.**np.arange(-8, 3, step=2))
    plt.plot(rd[0:-1], dn_dr)
    plt.show()

    # from fig. 1b
    assert 1e-3 < dn_dr[0] < 1e-2
    assert 1e1 < max(dn_dr) < 1e2
    assert dn_dr[-1] < 1e-9


def test_wet_vs_dry_spectrum(plot=False):
    # Arrange
    setup = Setup()

    # Act
    simulation = Simulation(setup)
    wet_volume = simulation.particles.state.get_backend_storage('x')
    wet_volume = simulation.particles.backend.to_ndarray(wet_volume)
    r_wet = Physics.x2r(wet_volume) / si.nanometre
    n = simulation.particles.backend.to_ndarray(simulation.particles.state.n)

    dry_volume = simulation.particles.state.get_backend_storage('dry volume')
    dry_volume = simulation.particles.backend.to_ndarray(dry_volume)
    r_dry = Physics.x2r(dry_volume) / si.nanometre

    # Plot
    if plot:
        plt.plot(r_wet, n)
        plt.plot(r_dry, n)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    # Assert
    assert (r_dry < r_wet).all()


def test_RH():
    # Arrange
    setup = Setup()

    # Act
    simulation = Simulation(setup)

    # Assert
    assert round(simulation.particles.environment["RH"][0], 3) == 0.856
