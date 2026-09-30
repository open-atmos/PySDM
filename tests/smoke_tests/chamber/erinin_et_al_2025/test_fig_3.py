"""tests supporting the
[Erinin et al. 2025](https://doi.org/10.48550/arXiv.2501.01467) example"""

from pathlib import Path
import numpy as np
import pytest
from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Erinin_et_al_2025
from PySDM.physics import si

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
