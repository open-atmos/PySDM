from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Matsushima_et_al_2023

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    return notebook_vars(
        Path(Matsushima_et_al_2023.__file__).parent / "figure_1.ipynb", plot=PLOT
    )


class TestFig1:
    @staticmethod
    def test_panel_b(notebook_local_variables):
        for alpha, y_data in notebook_local_variables["yas"].items():
            minimum_value = np.amin(y_data)
            print(alpha, minimum_value)
            assert minimum_value == 0 if alpha != 1 else 0.1
