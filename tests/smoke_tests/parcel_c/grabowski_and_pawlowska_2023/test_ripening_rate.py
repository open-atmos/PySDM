"""
test for values of the ripening rate
"""

import numpy as np
import pytest
from PySDM_examples.Grabowski_and_Pawlowska_2023 import Settings, Simulation

from PySDM.physics import si
from PySDM.products import RipeningRate

PRODUCTS = [
    RipeningRate(
        name="ripening",
    )
]

N_SD = 200

VELOCITIES_CM_PER_S = (25, 100)

AEROSOLS = ("pristine", "polluted")

DT = 1 * si.s

RTOL = 0.1


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
                    dt=DT,
                    vertical_velocity=vertical_velocity,
                    aerosol=aerosol,
                ),
                products=PRODUCTS,
            ).run()
    return outputs


@pytest.mark.parametrize("aerosol", AEROSOLS)
@pytest.mark.parametrize("w_cm_per_s", VELOCITIES_CM_PER_S)
def test_ripening_rate(
    outputs: dict,
    w_cm_per_s: int,
    aerosol: str,
):
    # arrange
    expected = {
        "pristine": {
            25: 0,
            100: 0,
        },
        "polluted": {
            25: 2.8 * 1e8,
            100: 4 * 1e8,
        },
    }[aerosol][w_cm_per_s]
    output = outputs[aerosol][w_cm_per_s]["products"]["ripening"]

    # assert
    assert np.isclose(
        np.max(output),
        expected,
        rtol=RTOL,
    ).all()
