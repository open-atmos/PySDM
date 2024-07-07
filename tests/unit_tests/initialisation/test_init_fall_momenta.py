"""
Test initialization of
`PySDM.attributes.physics.relative_fall_velocity.RelativeFallMomentum`
"""

import numpy as np
import pytest

from PySDM.builder import Builder
from PySDM.environments.box import Box
from PySDM.initialisation import init_fall_momenta
from PySDM.physics import si


@pytest.fixture(
    name="params",
    params=(
        pytest.param(
            {
                "multiplicity": np.array([1, 2, 3, 2]),
                "water mass": np.array(
                    [
                        1 * si.mg,
                        0.1 * si.mg,
                        1 * si.mg,
                        0.05 * si.mg,
                    ]
                ),
            },
        ),
    ),
)
def params_fixture(request):
    return request.param


class TestInitFallMomenta:
    @staticmethod
    def test_init_to_terminal_velocity(params, backend_instance):
        """
        Fall momenta correctly initialized to the terminal velocity * mass.
        """
        env = Box(dt=1, dv=1)
        builder = Builder(
            n_sd=len(params["multiplicity"]), backend=backend_instance, environment=env
        )
        builder.request_attribute("terminal velocity")
        particulator = builder.build(
            attributes={
                "multiplicity": params["multiplicity"],
                "water mass": params["water mass"],
            },
            products=(),
        )

        terminal_momentum = (
            particulator.attributes["terminal velocity"].to_ndarray()
            * params["water mass"]
        )

        assert np.allclose(init_fall_momenta(params["water mass"]), terminal_momentum)

    @staticmethod
    def test_init_to_zero(params):
        """
        Fall momenta correctly initialized to zero.
        """

        fall_momenta = init_fall_momenta(params["water mass"], zero=True)

        assert (fall_momenta == np.zeros_like(fall_momenta)).all()
