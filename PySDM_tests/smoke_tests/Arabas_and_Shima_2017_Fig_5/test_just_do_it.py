"""
Created at 2019
"""

from PySDM_examples.Arabas_and_Shima_2017_Fig_5.simulation import Simulation
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.setup import setups
from PySDM_tests.smoke_tests.utils import bdf

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection


rtols = [1e-3, 1e-7]
schemes = ['default', 'BDF']
setups_num = len(setups)


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
                setup.n_output = 20
                simulation = Simulation(setup)
                if scheme == 'BDF':
                    bdf.patch_core(simulation.core, setup.coord, rtol=1e-4)
                results = simulation.run()
                data[scheme][rtol].append(results)
    return data


def add_color_line(ax, x, y, z):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('plasma'), norm=plt.Normalize(-10, 0))
    z = np.log2(np.array(z))
    lc.set_array(z)
    lc.set_linewidth(3)

    ax.add_collection(lc)
    return lc


def test_plot(data, plot=False, save=False):
    if plot:
        fig, axs = plt.subplots(setups_num, len(rtols),
                                sharex=True, sharey=True, figsize=(10, 13))
        for setup_idx in range(setups_num):
            for rtol_idx in range(len(rtols)):
                ax = axs[setup_idx, rtol_idx]
                for scheme in schemes:
                    datum = data[scheme][rtols[rtol_idx]][setup_idx]
                    S = datum['S']
                    z = datum['z']
                    dt = datum['dt']
                    if scheme == 'BDF':
                        ax.plot(S, z, label=scheme, color='grey')
                    else:
                        lc = add_color_line(ax, S, z, dt)
                _rtol = '$r_{tol}$'
                ax.set_title(f"setup: {setup_idx}; {_rtol}: {rtols[rtol_idx]}")
                ax.set_xlim(-7.5e-3, 7.5e-3)
                ax.set_ylim(0, 180)
                ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        fig.colorbar(lc, ax=axs.flat, orientation='horizontal')

        if save:
            plt.savefig('ADAPTIVEvsBDF.pdf', format='pdf')
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
