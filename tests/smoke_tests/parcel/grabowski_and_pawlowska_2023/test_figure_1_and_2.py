"""
test against values read from plots in
[Grabowski and Pawlowska](https://doi.org/10.1029/2022GL101917) paper
"""

import numpy as np
import pytest
from PySDM_examples.Grabowski_and_Pawlowska_2023 import Settings, Simulation

from PySDM import Formulae
from PySDM.physics import si
from PySDM.products import AmbientRelativeHumidity

TRIVIA = Formulae().trivia

PRODUCTS = [
    AmbientRelativeHumidity(name="S_max", var="RH"),
]

N_SD = 25

VELOCITIES_CM_PER_S = (25, 100)

AEROSOLS = ("pristine", "polluted")

DZ = 500 * si.m

RTOL = 0.01


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


class TestFigure1And2:
    @staticmethod
    @pytest.mark.parametrize("aerosol", AEROSOLS)
    @pytest.mark.parametrize("w_cm_per_s", VELOCITIES_CM_PER_S)
    @pytest.mark.parametrize("attribute", ("volume", "equilibrium supersaturation"))
    @pytest.mark.parametrize("drop_id", (0, -1))
    def test_values_at_final_step(
        outputs: dict,
        w_cm_per_s: int,
        aerosol: str,
        attribute: str,
        drop_id: int,
    ):
        # arrange
        output = outputs[aerosol][w_cm_per_s]
        expected = {
            "volume": {
                "pristine": {
                    25: {
                        0: TRIVIA.volume(radius=0.04 * si.um),
                        -1: TRIVIA.volume(radius=20 * si.um),
                    },
                    100: {
                        0: TRIVIA.volume(radius=0.04 * si.um),
                        -1: TRIVIA.volume(radius=18 * si.um),
                    },
                },
                "polluted": {
                    25: {
                        0: TRIVIA.volume(radius=0.04 * si.um),
                        -1: TRIVIA.volume(radius=10 * si.um),
                    },
                    100: {
                        0: TRIVIA.volume(radius=0.04 * si.um),
                        -1: TRIVIA.volume(radius=10 * si.um),
                    },
                },
            },
            "equilibrium supersaturation": {
                "pristine": {
                    25: {0: 0.05 / 100 + 1, -1: 0.005 / 100 + 1},
                    100: {0: 0.15 / 100 + 1, -1: 0.005 / 100 + 1},
                },
                "polluted": {
                    25: {0: 0.025 / 100 + 1, -1: 0.005 / 100 + 1},
                    100: {0: 0.06 / 100 + 1, -1: 0.004 / 100 + 1},
                },
            },
        }[attribute][aerosol][w_cm_per_s][drop_id]

        # assert
        assert np.isclose(
            output["attributes"][attribute][drop_id][-1],
            expected,
            rtol=RTOL,
        ).all()

    @staticmethod
    @pytest.mark.parametrize("aerosol", AEROSOLS)
    @pytest.mark.parametrize("w_cm_per_s", VELOCITIES_CM_PER_S)
    @pytest.mark.parametrize(
        "rtol", (RTOL, pytest.param(RTOL / 100, marks=pytest.mark.xfail(strict=True)))
    )
    def test_ambient_humidity(
        outputs: dict,
        w_cm_per_s: int,
        aerosol: str,
        rtol: float,
    ):
        output = outputs[aerosol][w_cm_per_s]
        attributes = output["attributes"]
        for vol, crit_vol, eq_ss in zip(
            attributes["volume"],
            attributes["critical volume"],
            attributes["equilibrium supersaturation"],
        ):
            if np.all(vol < crit_vol):
                assert np.isclose(
                    output["products"]["S_max"], eq_ss, rtol=rtol, atol=0
                ).all()
