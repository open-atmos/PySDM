"""
test for condensation tolerance parameters: rtol_thd and rtol_x,
checking if supersaturation has more than one local maximum
"""

import numpy as np
import pytest
from PySDM_examples.Grabowski_and_Pawlowska_2023 import Settings, Simulation
from PySDM_examples.Grabowski_and_Pawlowska_2023.settings import condensation_tolerance
from scipy import signal

from PySDM.dynamics import condensation
from PySDM.physics import si
from PySDM.products import AmbientRelativeHumidity

PRODUCTS = [
    AmbientRelativeHumidity(name="S_max", var="RH"),
]

N_SD = 25
DZ = 10 * si.m
VELOCITIES_CM_PER_S = (25, 100, 400)
AEROSOLS = ("polluted",)


@pytest.mark.parametrize(
    "rtol_cond",
    (
        condensation_tolerance,
        pytest.param(
            condensation.DEFAULTS.rtol_thd, marks=pytest.mark.xfail(strict=True)
        ),
    ),
)
@pytest.mark.parametrize("aerosol", AEROSOLS)
@pytest.mark.parametrize("w_cm_per_s", VELOCITIES_CM_PER_S)
def test_condensation_tolerance(
    rtol_cond: float,
    w_cm_per_s: int,
    aerosol: str,
):
    # arrange
    output = Simulation(
        Settings(
            n_sd=25,
            dt=10 * si.s,
            vertical_velocity=w_cm_per_s * si.cm / si.s,
            aerosol=aerosol,
            rtol_thd=rtol_cond,
            rtol_x=rtol_cond,
        ),
        products=PRODUCTS,
    ).run()

    assert (
        signal.find_peaks(
            np.asarray(output["products"]["S_max"]),
            height=(1, None),
            prominence=(1e-5, None),
        )[0].shape[0]
        == 1
    )
