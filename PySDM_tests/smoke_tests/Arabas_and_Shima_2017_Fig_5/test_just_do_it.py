from PySDM_examples.Arabas_and_Shima_2017_Fig_5.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.setup import setups
from PySDM.simulation.dynamics.condensation import condensation
from PySDM_tests.smoke_tests.utils import bdf

import pytest
import numpy as np
import matplotlib.pyplot as plt


rtols = [1e-6, 1e-7]
schemes = ['default', 'BDF']
setups_num = len(setups)-4

@pytest.fixture(scope='module')
def data():
    # TODO: calculate BDF just once - as a reference solution
    data = {}
    for scheme in schemes:
        data[scheme] = {}
        for rtol in rtols:
            data[scheme][rtol] = []
            for setup_idx in range(setups_num):
                setup = setups[setup_idx]
                setup.scheme = scheme
                if scheme == 'default':
                    setup.rtol_x = rtol
                    setup.rtol_thd = rtol
                else:
                    setup.rtol_x = rtol
                    setup.rtol_thd = rtol
                setup.n_steps = 100
                simulation = Simulation(setup)
                if scheme == 'BDF':
                    bdf.patch_particles(simulation.particles, setup.coord, rtol=1e-4)
                data[scheme][rtol].append(simulation.run())
    return data


# @pytest.mark.skip
def test_plot(data, plot=False):
    if not plot:
        return
    fig, axs = plt.subplots(len(setups)-4, len(rtols),
                            sharex=True, sharey=True, figsize=(6, 15))
    for setup_idx in range(setups_num):
        for rtol_idx in range(len(rtols)):
            ax = axs[setup_idx, rtol_idx]
            for scheme in schemes:
                datum = data[scheme][rtols[rtol_idx]][setup_idx]
                S = datum['S']
                z = datum['z']
                ax.plot(S, z, label=scheme)
            ax.set_title(f"setup: {setup_idx}; rtol: {rtols[rtol_idx]}")
            ax.set_xlim(-.0075, .0075)
    ax.legend()
    plt.show()


def split(arg1, arg2):
    return arg1[0:np.argmax(arg2)+1], arg1[np.argmax(arg2):-1]


@pytest.mark.parametrize("setup_idx", range(setups_num))
@pytest.mark.parametrize("rtol", rtols)
@pytest.mark.parametrize("leg", ['ascent', 'descent'])
def test_vs_BDF(setup_idx, data, rtol, leg):
    # Arrange
    supersaturation = {}
    for scheme in schemes:
        sut = data[scheme][rtol][setup_idx]
        ascent, descent = split(sut['S'], sut['z'])
        supersaturation[scheme] = ascent if leg == 'ascent' else descent

    # Assert
    desired = np.array(supersaturation['BDF'])
    actual = np.array(supersaturation['default'])
    assert np.mean((desired - actual)**2) < rtol
    # np.testing.assert_allclose(
    #     desired=supersaturation['BDF'],
    #     actual=supersaturation['default'],
    #     rtol=rtol*1e3
    # )
