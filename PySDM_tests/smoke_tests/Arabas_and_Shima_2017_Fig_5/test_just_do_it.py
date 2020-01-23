from PySDM_examples.Arabas_and_Shima_2017_Fig_5.example import Simulation, setups
import pytest
import numpy as np
import matplotlib.pyplot as plt


ln10_dz_maxs = [-2, ]
schemes = ['BDF', 'libcloud']


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
                # TODO
                # setup.rtol = rtol
                # setup.atol = atol
                setup.dt_max = 10**ln10_dz_max / setup.w_avg
                print(setup.dt_max)
                setup.n_steps = 100
                data[scheme][ln10_dz_max].append(Simulation(setup).run())
    return data


def test_plot(data, plot=False):
    if not plot:
        return
    fig, axs = plt.subplots(len(setups), len(ln10_dz_maxs),
                            sharex=True, sharey=True, figsize=(6, 15))
    for setup_idx in range(len(setups)):
        for dt_idx in range(len(ln10_dz_maxs)):
            for scheme in schemes:
                datum = data[scheme][ln10_dz_maxs[dt_idx]][setup_idx]
                S = datum['S']
                z = datum['z']
                axs[setup_idx][dt_idx].plot(S, z)
#        plt.title(f"setup: {setup_idx}; dt_max: 10^{ln10_dz_max}; scheme: {scheme}")
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
        rtol=setups[setup_idx].rtol,
        atol=setups[setup_idx].atol
    )
