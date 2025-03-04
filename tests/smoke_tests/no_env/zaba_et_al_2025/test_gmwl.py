"""
test coefficients of global meteoric water line
"""

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
from sympy.abc import delta

from PySDM_examples import Zaba_et_al_2025
from PySDM.physics import si

PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    return notebook_vars(
        file=Path(Zaba_et_al_2025.__file__).parent / "global_meteoric_water_line.ipynb",
        plot=PLOT,
    )


@pytest.mark.parametrize("variant", ("VanHook", "HoritaAndWesolowski"))
def test_coeffs_plot(notebook_variables, variant):
    # arrange
    a = 8
    b = 10
    a_line = notebook_variables["a_line"][f"b={b}"]
    b_line = notebook_variables["b_line"][f"a={a}"]

    # act
    a_intersect = np.any(a_line > a) and np.any(a_line <= a)
    b_intersect = np.any(b_line > b) and np.any(b_line <= b)

    # assert
    np.testing.assert_equal(actual=a_intersect, desired=True)
    np.testing.assert_equal(actual=b_intersect, desired=True)
