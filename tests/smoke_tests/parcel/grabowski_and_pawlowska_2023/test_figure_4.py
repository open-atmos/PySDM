from PySDM_examples.Grabowski_and_Pawlowska_2023 import Settings, Simulation

from PySDM.physics import si
from PySDM.products import ActivatedMeanRadius, RadiusStandardDeviation

PRODUCTS = [
    ActivatedMeanRadius(name="r_act", count_activated=True, count_unactivated=False),
    RadiusStandardDeviation(
        name="r_std", count_activated=True, count_unactivated=False
    ),
]
import numpy as np


class TestFigure4:
    @staticmethod
    def test_pristine():
        # arrange
        output = {
            w_cm_per_s: Simulation(
                Settings(
                    vertical_velocity=w_cm_per_s * si.cm / si.s, aerosol="pristine"
                ),
                products=PRODUCTS,
            ).run()
            for w_cm_per_s in (25, 100, 400)
        }
        # act
        rel_dispersion = {
            key: output[key]["products"]["r_std"] / output[key]["products"]["r_act"]
            for key in output.keys()
        }

        # assert
        np.testing.assert_almost_equal(rel_dispersion[25][-1], 0.01, decimal=3)

    @staticmethod
    def test_polluted():
        # arrange
        pass

        # act

        # assert
