"""
checking consistency with values in the paper for Figure 2
"""

from pathlib import Path

import numpy as np
import pytest

from PySDM_examples.utils import notebook_vars
from PySDM_examples import Abade_and_Albuquerque_2024

from PySDM.physics import si


PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Abade_and_Albuquerque_2024.__file__).parent / "fig_2.ipynb", plot=PLOT
    )


class TestFig2:
    @staticmethod
    @pytest.mark.parametrize("key", ("total", "water"))
    def test_cloud_base(variables, key):
        height = np.asarray(variables["output"]["height"])
        assert (
            variables["values"][key][height < 0.9 * si.km] < 0.01 * si.g / si.kg
        ).all()
        assert (
            variables["values"][key][height > 1.1 * si.km] > 0.05 * si.g / si.kg
        ).all()

    @staticmethod
    @pytest.mark.parametrize(
        "var_name, desired_value",
        (
            ("total", 1.1 * si.g / si.kg),
            ("ice", 0.20 * si.g / si.kg),
            ("water", 0.93 * si.g / si.kg),
        ),
    )
    def test_values_at_cloud_top(variables, var_name, desired_value):
        np.testing.assert_approx_equal(
            desired=desired_value,
            actual=variables["values"][var_name][-1],
            significant=2,
        )

    @staticmethod
    @pytest.mark.parametrize("key", ("total", "ice", "water"))
    def test_monotonicity(variables, key):
        assert (np.diff(variables["values"][key]) > 0).all()
