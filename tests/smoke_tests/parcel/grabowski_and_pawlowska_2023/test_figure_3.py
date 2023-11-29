"""
test against values read from plots in
[Grabowski and Pawlowska](https://doi.org/10.1029/2022GL101917) paper
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

N_SD = 25

VELOCITIES_CM_PER_S = (25, 100, 400)

AEROSOLS = ("pristine", "polluted")

DZ = 500 * si.m
RTOL = 0.05


@pytest.fixture(scope="session", name="outputs")
def outputs_fixture():
    outputs = {}
    for aerosol in AEROSOLS:
        outputs[aerosol] = {}
        for w_cm_per_s in VELOCITIES_CM_PER_S:
            vertical_velocity = w_cm_per_s * si.cm / si.s
            outputs[aerosol][w_cm_per_s] = Simulation(
                Settings(
                    n_sd=N_SD,
                    dt=DZ / vertical_velocity,
                    vertical_velocity=vertical_velocity,
                    aerosol=aerosol,
                ),
                products=PRODUCTS,
            ).run()
    return outputs


@pytest.mark.parametrize("w_cm_per_s", VELOCITIES_CM_PER_S)
@pytest.mark.parametrize("aerosol", AEROSOLS)
@pytest.mark.parametrize("product", ("r_vol", "n_act", "area_std"))
@pytest.mark.parametrize(
    "rtol", (RTOL, pytest.param(RTOL / 500, marks=pytest.mark.xfail(strict=True)))
)
def test_values_at_final_step(
    outputs: dict, w_cm_per_s: int, aerosol: str, product: str, rtol: float
):
    # arrange
    output = outputs[aerosol][w_cm_per_s]
    expected = {
        "r_vol": {
            "pristine": {25: 20 * si.um, 100: 18 * si.um, 400: 15 * si.um},
            "polluted": {25: 10 * si.um, 100: 9 * si.um, 400: 9 * si.um},
        },
        "n_act": {
            "pristine": {
                25: 60 * si.cm**-3,
                100: 100 * si.cm**-3,
                400: 180 * si.cm**-3,
            },
            "polluted": {
                25: 350 * si.cm**-3,
                100: 550 * si.cm**-3,
                400: 550 * si.cm**-3,
            },
        },
        "area_std": {
            "pristine": {
                25: 8 * si.um**2 * 4 * np.pi,
                100: 10 * si.um**2 * 4 * np.pi,
                400: 6.5 * si.um**2 * 4 * np.pi,
            },
            "polluted": {
                25: 11 * si.um**2 * 4 * np.pi,
                100: 6 * si.um**2 * 4 * np.pi,
                400: 2.5 * si.um**2 * 4 * np.pi,
            },
        },
    }[product]

    # assert
    assert np.isclose(
        np.log(output["products"][product][-1]),
        np.log(expected[aerosol][w_cm_per_s]),
        rtol=rtol,
        atol=0,
    ).all()
