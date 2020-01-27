from PySDM_examples.Arabas_and_Shima_2017_Fig_5.example import Simulation, setups
from PySDM.simulation.dynamics.condensation import condensation
import pytest
import numpy as np
import matplotlib.pyplot as plt


ln10_dz_maxs = [-2.5, -3]
schemes = ['BDF', 'libcloud']

local_rtol = condensation.default_rtol
local_atol = condensation.default_atol
global_rtol = 10 * local_rtol
global_atol = 10 * local_atol


@pytest.fixture(scope='module')
def data():
    data = {}
    for scheme in schemes:
        data[scheme] = {}
        for ln10_dz_max in ln10_dz_maxs:
            data[scheme][ln10_dz_max] = []
            for setup_idx in range(len(setups)):
                setup = setups[setup_idx]
                setup.scheme = scheme
                setup.rtol = local_rtol
                setup.atol = local_atol
                setup.dt_max = 10**ln10_dz_max / setup.w_avg
                setup.n_steps = 100
                data[scheme][ln10_dz_max].append(Simulation(setup).run())
    return data


def test_plot(data, plot=True):
    if not plot:
        return
    fig, axs = plt.subplots(len(setups), len(ln10_dz_maxs),
                            sharex=True, sharey=True, figsize=(6, 15))
    for setup_idx in range(len(setups)):
        for dt_idx in range(len(ln10_dz_maxs)):
            ax = axs[setup_idx, dt_idx]
            for scheme in schemes:
                datum = data[scheme][ln10_dz_maxs[dt_idx]][setup_idx]
                S = datum['S']
                z = datum['z']
                ax.plot(S, z, label=scheme)
            ax.set_title(f"setup: {setup_idx}; dz_max: 10^{ln10_dz_maxs[dt_idx]}")
    ax.legend()
    plt.show()


def split(arg1, arg2):
    return arg1[0:np.argmax(arg2)+1], arg1[np.argmax(arg2):-1]


@pytest.mark.parametrize("setup_idx", range(len(setups)))
@pytest.mark.parametrize("ln10_dz_max", ln10_dz_maxs)
@pytest.mark.parametrize("leg", ['ascent', 'descent'])
def test_vs_BDF(setup_idx, data, ln10_dz_max, leg):
    # Arrange
    supersaturation = {}
    for scheme in schemes:
        sut = data[scheme][ln10_dz_max][setup_idx]
        ascent, descent = split(sut['S'], sut['z'])
        supersaturation[scheme] = ascent if leg == 'ascent' else descent

    # Assert
    np.testing.assert_allclose(
        desired=supersaturation['BDF'],
        actual=supersaturation['libcloud'],
        rtol=global_rtol,
        atol=global_atol
    )
