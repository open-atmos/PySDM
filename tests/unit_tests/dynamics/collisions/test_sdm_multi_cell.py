# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.backends import ThrustRTC
from PySDM.dynamics.collisions.collision import DEFAULTS
from PySDM.environments import Box
from PySDM.impl.mesh import Mesh
from PySDM.initialisation.sampling.spatial_sampling import Pseudorandom

from .conftest import get_dummy_particulator_and_coalescence


class TestSDMMultiCell:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize("n_sd", [2, 3, 8000])
    @pytest.mark.parametrize("adaptive", [False, True])
    def test_coalescence_call(n_sd, backend_class, adaptive):
        if backend_class is ThrustRTC:
            pytest.skip("TODO #330")

        # Arrange
        n = np.ones(n_sd)
        v = np.ones_like(n)
        env = Box(dv=1, dt=DEFAULTS.dt_coal_range[1])
        grid = (25, 25)
        env.mesh = Mesh(grid, size=grid)
        particulator, sut = get_dummy_particulator_and_coalescence(
            backend_class, len(n), environment=env
        )
        cell_id, _, _ = env.mesh.cellular_attributes(
            Pseudorandom.sample(backend=particulator.backend, grid=grid, n_sd=len(n))
        )
        attributes = {"multiplicity": n, "volume": v, "cell id": cell_id}
        particulator.build(attributes)
        sut.actual_length = particulator.attributes._ParticleAttributes__idx.length
        sut.adaptive = adaptive

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(
            cell_id, particulator.attributes["cell id"].to_ndarray(raw=True)
        )
