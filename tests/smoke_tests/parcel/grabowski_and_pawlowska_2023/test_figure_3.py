"""
test against values read from plots in Grabowski and Pawlowska paper
"""

import numpy as np
import pytest
from PySDM_examples.Grabowski_and_Pawlowska_2023 import Settings, Simulation

from PySDM.physics import si
from PySDM.products import (
    ActivatedParticleConcentration,
    AreaStandardDeviation,
    MeanVolumeRadius,
)

PRODUCTS = [
    ActivatedParticleConcentration(
        name="n_act", count_activated=True, count_unactivated=False, stp=True
    ),
    MeanVolumeRadius(name="r_vol", count_activated=True, count_unactivated=False),
    AreaStandardDeviation(
        name="area_std", count_activated=True, count_unactivated=False
    ),
]

COMMON_SETTINGS = {"dt": 1 * si.s, "n_sd": 200}


@pytest.mark.parametrize("w_cm_per_s", (25, 100, 400))
@pytest.mark.parametrize("aerosol", ("pristine", "polluted"))
def test_pristine(w_cm_per_s: int, aerosol: str):
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
    r_vol_mean = output["products"]["r_vol"]
    n_act = output["products"]["n_act"]
    area_std = np.asarray(output["products"]["area_std"]) / (4 * np.pi)

    # assert
    assert np.isclose(
        r_vol_mean[-1],
        {
            "pristine": {25: 11 * si.um, 100: 10.5 * si.um, 400: 10 * si.um},
            "polluted": {25: 10 * si.um, 100: 9 * si.um, 400: 9 * si.um},
        }[aerosol][w_cm_per_s],
        rtol=0.2,
    ).all()
    assert np.isclose(
        n_act[-1],
        {
            "pristine": {
                25: 50 * si.cm**-3,
                100: 100 * si.cm**-3,
                400: 200 * si.cm**-3,
            },
            "polluted": {
                25: 350 * si.cm**-3,
                100: 550 * si.cm**-3,
                400: 550 * si.cm**-3,
            },
        }[aerosol][w_cm_per_s],
        rtol=0.4,
    ).all()
    assert np.isclose(
        area_std[-1],
        {
            "pristine": {
                25: 8 * si.um**2,
                100: 10 * si.um**2,
                400: 6.5 * si.um**2,
            },
            "polluted": {
                25: 11 * si.um**2,
                100: 6 * si.um**2,
                400: 2.5 * si.um**2,
            },
        }[aerosol][w_cm_per_s],
        rtol=0.5,
    ).all()

    # @staticmethod
    # @pytest.mark.parametrize("w_cm_per_s", (25, 100, 400))
    # def test_polluted(w_cm_per_s):
    #     # arrange
    #     output = Simulation(
    #         Settings(
    #             **COMMON_SETTINGS,
    #             vertical_velocity=w_cm_per_s * si.cm / si.s,
    #             aerosol="polluted"
    #         ),
    #         products=PRODUCTS,
    #     ).run()
    #
    #     r_vol_mean = output["products"]["r_vol"]
    #     n_act = output["products"]["n_act"]
    #     area_std = np.asarray(output["products"]["area_std"])/(4*np.pi)
    #
    #     # assert
    #     assert np.isclose(
    #         r_vol_mean[-1],
    #         {25: 10*si.um, 100: 9*si.um, 400: 9*si.um}[w_cm_per_s],
    #         rtol=0.15,
    #     ).all()
    #     assert np.isclose(
    #         n_act[-1],
    #         {25: 350 * si.cm**-3, 100: 550 * si.cm**-3, 400: 550 * si.cm**-3}[w_cm_per_s],
    #         rtol=0.1,
    #     ).all()
    #     assert np.isclose(
    #         area_std[-1],
    #         {25: 11 * si.um**2, 100: 6 * si.um**2, 400: 2.5 * si.um**2}[w_cm_per_s],
    #         rtol=0.1,
    #     ).all()
