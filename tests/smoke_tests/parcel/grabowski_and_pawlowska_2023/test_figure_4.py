from PySDM_examples.Grabowski_and_Pawlowska_2023 import Settings, Simulation

from PySDM.physics import si
from PySDM.products import ActivatedMeanRadius, RadiusStandardDeviation

PRODUCTS = [
    ActivatedMeanRadius(name="r_act", count_activated=True, count_unactivated=False),
    RadiusStandardDeviation(
        name="r_std", count_activated=True, count_unactivated=False
    ),
]

COMMON_SETTINGS = {"dt": 5 * si.s, "n_sd": 20}

import numpy as np
import pytest


class TestFigure4:
    @staticmethod
    @pytest.mark.parametrize("w_cm_per_s", (25, 100, 400))
    def test_pristine(w_cm_per_s: int):
        # arrange
        output = Simulation(
            Settings(
                **COMMON_SETTINGS,
                vertical_velocity=w_cm_per_s * si.cm / si.s,
                aerosol="pristine"
            ),
            products=PRODUCTS,
        ).run()

        # act
        rel_dispersion = {
            key: np.asarray(output["products"]["r_std"])
            / np.asarray(output["products"]["r_act"])
            for key in output.keys()
        }

        # assert
        np.testing.assert_almost_equal(
            actual=rel_dispersion[-1],
            desired={25: 0.01, 100: 0.02, 400: 0.015}[w_cm_per_s],
            decimal=2,
        )

    @staticmethod
    @pytest.mark.parametrize("w_cm_per_s", (25, 100, 400))
    def test_polluted(w_cm_per_s):
        # arrange
        output = Simulation(
            Settings(
                **COMMON_SETTINGS,
                vertical_velocity=w_cm_per_s * si.cm / si.s,
                aerosol="polluted"
            ),
            products=PRODUCTS,
        ).run()

        # act
        rel_dispersion = {
            key: np.asarray(output[key]["products"]["r_std"])
            / np.asarray(output[key]["products"]["r_act"])
            for key in output.keys()
        }

        # assert
        np.testing.assert_almost_equal(
            actual=rel_dispersion[-1],
            desired={25: 0.09, 100: 0.03, 400: 0.015}[w_cm_per_s],
            decimal=2,
        )
