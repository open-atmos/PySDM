"""
test against values read from plots in Grabowski and Pawlowska paper
"""

import numpy as np
import pytest
from PySDM_examples.Grabowski_and_Pawlowska_2023 import Settings, Simulation

from PySDM.physics import si
from PySDM.products import ActivatedMeanRadius, RadiusStandardDeviation

PRODUCTS = [
    ActivatedMeanRadius(name="r_act", count_activated=True, count_unactivated=False),
    RadiusStandardDeviation(
        name="r_std", count_activated=True, count_unactivated=False
    ),
]
COMMON_SETTINGS = {"dt": 1 * si.s, "n_sd": 100}


@pytest.mark.parametrize("w_cm_per_s", (25, 100, 400))
@pytest.mark.parametrize("aerosol", ("pristine", "polluted"))
def test(w_cm_per_s: int, aerosol: str):
    # arrange
    output = Simulation(
        Settings(
            **COMMON_SETTINGS,
            vertical_velocity=w_cm_per_s * si.cm / si.s,
            aerosol=aerosol
        ),
        products=PRODUCTS,
    ).run()

    # act
    rel_dispersion = np.asarray(output["products"]["r_std"]) / np.asarray(
        output["products"]["r_act"]
    )

    # assert
    # np.testing.assert_almost_equal(
    #     actual=rel_dispersion[-1],
    #     desired={"pristine":{25: 0.01, 100: 0.02, 400: 0.015},
    #              "polluted":{25: 0.09, 100: 0.03, 400: 0.015}
    #              }[aerosol][w_cm_per_s],
    #     decimal=2,
    # )

    assert np.isclose(
        rel_dispersion[-1],
        {
            "pristine": {25: 0.01, 100: 0.02, 400: 0.015},
            "polluted": {25: 0.09, 100: 0.03, 400: 0.015},
        }[aerosol][w_cm_per_s],
        rtol=0.7,
    ).all()
