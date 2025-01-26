from pathlib import Path
import numpy as np
import pytest
from open_atmos_jupyter_utils import notebook_vars
from PySDM.physics import si
from PySDM_examples import Erinin_et_al_2025

PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    return notebook_vars(
        file=Path(Erinin_et_al_2025.__file__).parent / "fig_3.ipynb",
        plot=PLOT,
    )


class TestFig3:
    @staticmethod
    def test_final_pressure(notebook_variables):
        assert (
            0.45 * si.bar
            < np.amin(notebook_variables["output"]["ambient pressure"])
            < 0.46 * si.bar
        )

    @staticmethod
    def test_final_temperature(notebook_variables):
        assert (
            258 * si.K
            < np.amin(notebook_variables["output"]["ambient temperature"])
            < 259 * si.K
        )
