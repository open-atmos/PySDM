"""
test coefficients of global meteoric water line
"""

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars

from PySDM_examples import Zaba_et_al_2025

PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    return notebook_vars(
        file=Path(Zaba_et_al_2025.__file__).parent / "global_meteoric_water_line.ipynb",
        plot=PLOT,
    )


@pytest.mark.parametrize(
    "variant",
    (
        pytest.param("Majoube1971"),
        pytest.param("HoritaAndWesolowski1994"),
        pytest.param("VanHook1968", marks=pytest.mark.xfail(strict=True)),
    ),
)
@pytest.mark.parametrize("parameter", (("b=10", 8), ("a=8", 10)))
def test_plot(notebook_variables, variant, parameter):
    """test if lines for a = 8 and b = 10 intersect"""
    # arrange
    line = notebook_variables["lines"][variant][parameter[0]]

    # act
    if_intersect = np.any(line > parameter[1]) and np.any(line <= parameter[1])

    # assert
    np.testing.assert_equal(actual=if_intersect, desired=True)
