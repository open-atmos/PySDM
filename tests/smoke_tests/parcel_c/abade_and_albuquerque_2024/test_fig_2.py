"""
checking consistency with values in the paper for Figure 2
"""

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
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
    @pytest.mark.parametrize(
        "model, key",
        (
            ("Bulk", "total"),
            ("Bulk", "water"),
            ("Homogeneous", "total"),
        ),
    )
    def test_cloud_base(variables, key, model):
        height = np.asarray(variables["output"][model]["height"])
        assert (
            variables["values"][model][key][height < 0.9 * si.km] < 0.01 * si.g / si.kg
        ).all()
        assert (
            variables["values"][model][key][height > 1.1 * si.km] > 0.05 * si.g / si.kg
        ).all()

    @staticmethod
    @pytest.mark.parametrize(
        "model, var_name, desired_value",
        (
            ("Bulk", "total", 1.1 * si.g / si.kg),
            ("Bulk", "ice", 0.16 * si.g / si.kg),
            ("Bulk", "water", 0.90 * si.g / si.kg),
            ("Homogeneous", "total", 1.1 * si.g / si.kg),
            ("Homogeneous", "ice", 1.1 * si.g / si.kg),
            ("Homogeneous", "water", 2.9e-9),
        ),
    )
    def test_values_at_cloud_top_for(variables, model, var_name, desired_value):
        np.testing.assert_approx_equal(
            desired=desired_value,
            actual=variables["values"][model][var_name][-1],
            significant=2,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "model, key",
        (
            ("Homogeneous", "total"),
            ("Homogeneous", "ice"),
            ("Bulk", "total"),
            ("Bulk", "ice"),
            ("Bulk", "water"),
        ),
    )
    def test_monotonicity(variables, model, key):
        assert (np.diff(variables["values"][model][key]) >= 0).all()
