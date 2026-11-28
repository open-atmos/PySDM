import numpy as np
import pytest
from PySDM import Particulator
from PySDM.physics import si
from PySDM.dynamics import Coalescence
from PySDM.dynamics.collisions.collision_kernels.constantK import ConstantK
from PySDM.environments import Box


@pytest.mark.parametrize(
    "kernel_const, multiplicities, tf_in, tf_out",
    (
        (0, [2, 2], [250 * si.K, 200 * si.K], [250 * si.K, 200 * si.K]),
        (0, [2, 2], [200 * si.K, 250 * si.K], [200 * si.K, 250 * si.K]),
        (1, [1, 1], [250 * si.K, 200 * si.K], [250 * si.K]),
        (1, [1, 1], [200 * si.K, 250 * si.K], [250 * si.K]),
        (0.5, [2, 2], [250 * si.K, 200 * si.K], [250 * si.K, 250 * si.K]),
        (0.5, [2, 2], [200 * si.K, 250 * si.K], [250 * si.K, 250 * si.K]),
        (1, [2, 1], [200 * si.K, 250 * si.K], [250 * si.K]),
        (1, [1, 2], [200 * si.K, 250 * si.K], [250 * si.K]),
        (0.5, [2, 1], [200 * si.K, 250 * si.K], [200 * si.K, 250 * si.K]),
        (0.5, [1, 2], [200 * si.K, 250 * si.K], [250 * si.K, 250 * si.K]),
    ),
)
def test_maximum_attribute(
    backend_instance, kernel_const, multiplicities, tf_in, tf_out
):
    # Arrange
    attr = "freezing temperature"

    particulator = Particulator(
        dynamics=(
            Coalescence(collision_kernel=ConstantK(a=kernel_const), adaptive=False),
        ),
        environment=Box(dt=1, dv=1, backend=backend_instance),
        attributes={
            attr: np.asarray(tf_in),
            "multiplicity": np.asarray(multiplicities),
            "water mass": np.asarray([np.nan, np.nan]) * si.ug,
        },
        n_sd=2,
    )

    # Act
    assert particulator.attributes.super_droplet_count == 2
    particulator.advance(n_steps=1)

    # Assert
    assert all(particulator.attributes[attr].to_ndarray(raw=True) == np.asarray(tf_out))
