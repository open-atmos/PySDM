from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from PySDM_examples.utils import notebook_vars
from PySDM_examples import Jensen_Nugent_2016
from PySDM.physics.constants import PER_CENT

from PySDM.physics import si

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Jensen_Nugent_2016.__file__).parent / "Fig_3.ipynb", plot=PLOT
    )


class TestFig3:
    @staticmethod
    def test_cloud_base_height(variables):
        supersaturation = variables["output"]["products"]["S_max"]
        for index, value in enumerate(supersaturation):
            if value > 0:
                cloud_base_index = index
                break

        z0 = variables["settings"].z0
        assert (
            290 * si.m
            < variables["output"]["products"]["z"][cloud_base_index] - z0
            < 300 * si.m
        )

    @staticmethod
    def test_supersaturation_maximum(variables):
        supersaturation = np.asarray(variables["output"]["products"]["S_max"])
        assert signal.argrelextrema(supersaturation, np.greater)[0].shape[0] == 1
        assert 0.35 * PER_CENT < np.nanmax(supersaturation) < 0.5 * PER_CENT


# TODO: asserts on radii
