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
UPDRAFTS = (3.6, 0.4)
N_SD = 64


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
            *[(f"Bulk-{updraft}", "water") for updraft in UPDRAFTS],
            *[
                (f"Homogeneous-{im_freeze}-{N_SD}-{updraft}", "water")
                for updraft in UPDRAFTS
                for im_freeze in ("ABIFM", "INAS")
            ],
        ),
    )
    def test_cloud_base(variables, key, model):
        data = variables["datasets"][model]["realisations"][0]
        height = np.asarray(data["height"])
        assert (np.asarray(data[key])[height < 0.9 * si.km] < 0.01 * si.g / si.kg).all()
        assert (np.asarray(data[key])[height > 1.1 * si.km] > 0.05 * si.g / si.kg).all()

    @staticmethod
    @pytest.mark.parametrize(
        "model, var_name, desired_value",
        (
            ("Bulk-3.6", "ice", 0.16 * si.g / si.kg),
            ("Bulk-3.6", "water", 0.90 * si.g / si.kg),
            ("Homogeneous-INAS-64-3.6", "ice", 1.1 * si.g / si.kg),
            ("Homogeneous-INAS-64-3.6", "water", 2.9e-9),
            ("Homogeneous-ABIFM-64-3.6", "ice", 1.1 * si.g / si.kg),
            ("Homogeneous-ABIFM-64-3.6", "water", 2.6e-9),
        ),
    )
    def test_values_at_cloud_top(variables, model, var_name, desired_value):
        np.testing.assert_approx_equal(
            desired=desired_value,
            actual=variables["datasets"][model]["realisations"][0][var_name][-1],
            significant=2,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "model, key",
        [
            *[
                (f"Bulk-{updraft}", key)
                for updraft in UPDRAFTS
                for key in ("water", "ice", "vapour")
            ],
            *[
                (f"Homogeneous-{im_freeze}-{N_SD}-{updraft}", key)
                for updraft in UPDRAFTS
                for im_freeze in ("ABIFM", "INAS")
                for key in ("water", "ice", "vapour", "ice+water", "total")
            ],
        ],
    )
    def test_monotonicity(variables, model, key):
        delta_mixing_ratio = np.diff(
            variables["datasets"][model]["realisations"][0][key]
        )
        if key in ("water", "ice", "ice+water"):
            assert (delta_mixing_ratio >= 0).all()
            assert (delta_mixing_ratio > 0).any()
        elif key == "vapour":
            assert (delta_mixing_ratio <= 0).all()
            assert (delta_mixing_ratio < 0).any()
        elif key == "total":
            assert (delta_mixing_ratio == 0).all()
