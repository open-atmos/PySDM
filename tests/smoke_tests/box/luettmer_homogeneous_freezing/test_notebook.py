"""basic tests for threshold freezing temperatures"""

from pathlib import Path
import numpy as np

import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Luettmer_homogeneous_freezing
from PySDM.physics.constants_defaults import HOMOGENEOUS_FREEZING_THRESHOLD

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Luettmer_homogeneous_freezing.__file__).parent
        / "simple_homogenous_freezing_example.ipynb",
        plot=PLOT,
    )

@staticmethod
def test_freezing_temperatures(variables):
    for simulation in variables["simulations"]:
        output = simulation["ensemble_member_outputs"][0]
        T_frz = np.asarray(output["T_frz"])

        # assert
        assert all(np.isfinite(T_frz))
        assert all(T_frz > 0)
        assert all(T_frz - HOMOGENEOUS_FREEZING_THRESHOLD > -0.5)
