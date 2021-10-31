from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM_examples.Bartman_2020_MasterThesis.fig_5_BDF_VS_ADAPTIVE import data as data_method

import pytest
import numpy as np


rtols = (1e-3, 1e-7)
schemes = ('CPU', 'GPU', 'BDF')
setups_num = len(setups)


@pytest.fixture(scope='module')
def data():
    return data_method(n_output=20, rtols=rtols, schemes=schemes, setups_num=setups_num)


def split(arg1, arg2):
    return arg1[0:np.argmax(arg2)+1], arg1[np.argmax(arg2):-1]


@pytest.mark.parametrize("settings_idx", range(setups_num))
@pytest.mark.parametrize("rtol", rtols)
@pytest.mark.parametrize("leg", ['ascent', 'descent'])
@pytest.mark.parametrize("scheme", ('CPU', 'GPU'))
def test_vs_BDF(settings_idx, data, rtol, leg, scheme):
    # Arrange
    supersaturation = {}
    for scheme in schemes:
        sut = data[scheme][rtol][settings_idx]
        ascent, descent = split(sut['S'], sut['z'])
        supersaturation[scheme] = ascent if leg == 'ascent' else descent

    # Assert
    desired = np.array(supersaturation['BDF'])
    actual = np.array(supersaturation[scheme])
    assert np.mean((desired - actual)**2) < rtol
