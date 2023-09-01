"""
test against values read from plots in
[Grabowski and Pawlowska 2023](https://doi.org/10.1029/2022GL101917) paper
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
DZ = 250 * si.m
N_SD = 32
RTOL = 0.3


@pytest.mark.parametrize("w_cm_per_s", (25, 100, 400))
@pytest.mark.parametrize("aerosol", ("pristine", "polluted"))
def test_values_at_final_step(w_cm_per_s: int, aerosol: str):
    # arrange
    vertical_velocity = w_cm_per_s * si.cm / si.s
    output = Simulation(
        Settings(
            n_sd=N_SD,
            dt=DZ / vertical_velocity,
            vertical_velocity=vertical_velocity,
            aerosol=aerosol,
        ),
        products=PRODUCTS,
    ).run()

    # act
    rel_dispersion_at_final_step = np.asarray(
        output["products"]["r_std"][-1]
    ) / np.asarray(output["products"]["r_act"][-1])

    # assert
    assert np.isclose(
        np.log(rel_dispersion_at_final_step),
        np.log(
            {
                "pristine": {25: 0.01, 100: 0.02, 400: 0.015},
                "polluted": {25: 0.09, 100: 0.03, 400: 0.015},
            }[aerosol][w_cm_per_s]
        ),
        rtol=RTOL,
        atol=0,
    ).all()
