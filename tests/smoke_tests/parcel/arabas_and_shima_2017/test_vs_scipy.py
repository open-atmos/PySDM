# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM_examples.Bartman_2020_MasterThesis.fig_5_SCIPY_VS_ADAPTIVE import (
    data as data_method,
)

rtols = (1e-2,)
schemes = ("CPU", "SciPy")  # ,'GPU')  # TODO #588
setups_num = len(setups)


@pytest.fixture(scope="module", name="data")
def data_fixture():
    return data_method(n_output=20, rtols=rtols, schemes=schemes, setups_num=setups_num)


def split(arg1, arg2):
    return arg1[0 : np.argmax(arg2) + 1], arg1[np.argmax(arg2) : -1]


@pytest.mark.parametrize("settings_idx", range(setups_num))
@pytest.mark.parametrize("rtol", rtols)
@pytest.mark.parametrize("leg", ["ascent", "descent"])
@pytest.mark.parametrize("scheme", ("CPU",))  # 'GPU'))  # TODO #588
def test_vs_scipy(settings_idx, data, rtol, leg, scheme):
    # Arrange
    supersaturation = {}
    for sch in schemes:
        sut = data[sch][rtol][settings_idx]
        ascent, descent = split(sut["S"], sut["z"])
        supersaturation[sch] = ascent if leg == "ascent" else descent

    # Assert
    desired = np.array(supersaturation["SciPy"])
    actual = np.array(supersaturation[scheme])
    assert np.mean((desired - actual) ** 2) < rtol
