from examples.Yang_et_al_2018.example import Simulation, Setup
from PySDM.simulation.physics.constants import si
from PySDM.utils import Physics
import matplotlib.pyplot as plt
import numpy as np


def test_spectrum_x():
    simulation = Simulation()
    dry_volume = simulation.particles.state.get_backend_storage('x') #TODO: to_ndarray
    rd = Physics.x2r(dry_volume) / si.nanometre

    rd = rd[::-1]
    assert round(rd[  1-1], 0) == 503
    assert round(rd[ 10-1], 0) == 355
    assert round(rd[ 50-1], 1) == 75.3
    assert round(rd[100-1], 1) == 10.8


def test_spectrum_y():
    simulation = Simulation()
    dry_volume = simulation.particles.state.get_backend_storage('x') #TODO: to_ndarray
    rd = Physics.x2r(dry_volume) / si.nanometre
    nd = simulation.particles.state.n # TODO: to_ndarray

    dr = (rd[1:] - rd[0:-1]) / si.nanometre
    env = simulation.particles.environment
    dn_dr = (nd[0:-1] / env.mass * env.rho / dr)
    dn_dr /= (1/si.centimetre**3)

    plt.figure(figsize=(5,5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e1, 1e3)
    plt.ylim(1e-9, 1e3)
    plt.yticks(10.**np.arange(-8, 3, step=2))
    plt.plot(rd[0:-1], dn_dr)
    plt.show()

    # from fig. 1b
    assert 1e-3 < dn_dr[0] < 1e-1
    assert 1e1 < max(dn_dr) < 1e2
    assert dn_dr[-1] < 1e-9

