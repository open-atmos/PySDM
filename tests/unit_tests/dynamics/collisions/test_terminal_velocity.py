from contextlib import nullcontext

import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.environments import Box
from PySDM.physics import si


@pytest.mark.parametrize(
    "variant, water_mass, exception_context, expected_v_term",
    (
        ("GunnKinzer1949", 0 * si.g, None, 0),
        ("GunnKinzer1949", 1e10 * si.kg, pytest.raises(ValueError, match="Radii"), -1),
        ("RogersYau", 0 * si.g, None, 0),
        ("TpDependent", 0 * si.g, None, 0),
    ),
)
def test_terminal_velocity(
    variant, backend_class, water_mass, exception_context, expected_v_term
):
    if variant == "TpDependent":
        pytest.skip()  # TODO #348

    # arrange
    if exception_context is None:
        context = nullcontext()
    else:
        context = exception_context

    formulae = Formulae(terminal_velocity=variant)
    builder = Builder(n_sd=1, backend=backend_class(formulae))
    builder.set_environment(Box(dv=np.nan, dt=np.nan))
    builder.request_attribute("terminal velocity")
    particulator = builder.build(
        attributes={
            "water mass": np.asarray([water_mass]),
            "multiplicity": np.asarray([-1]),
        }
    )

    # act
    with context:
        v_term = particulator.attributes["terminal velocity"].to_ndarray()

        # assert
        np.testing.assert_approx_equal(v_term, expected_v_term)
