"""
test checking values on Fig 1
"""

from pathlib import Path
import pytest
import numpy as np

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Matsushima_et_al_2023

PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    """returns variables from the notebook Matsushima_et_al/figure_1.ipynb"""
    return notebook_vars(
        file=Path(Matsushima_et_al_2023.__file__).parent / "figure_1.ipynb", plot=PLOT
    )


def test_fig(notebook_variables):
    """test if ventilation coefficients (f) have the same line slope"""
    alpha = 0
    print(notebook_variables["plot_a"][alpha])


