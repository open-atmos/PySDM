"""
Created at 2019
"""

from PySDM_examples.Arabas_and_Shima_2017_Fig_5.setup import setups
from PySDM_examples.MasterThesis.fig_5_BDF_VS_ADAPTIVE import plot as plot_method
from PySDM_examples.MasterThesis.fig_5_BDF_VS_ADAPTIVE import data as data_method

import pytest
import numpy as np


rtols = [1e-3, 1e-7]
schemes = ['default', 'BDF']
setups_num = len(setups)


@pytest.fixture(scope='module')
def data():
    return data_method(n_output=20, rtols=rtols, schemes=schemes, setups_num=setups_num)


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
