# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.backends import CPU, GPU, ThrustRTC
from PySDM.backends.impl_common.index import make_Index
from PySDM.backends.impl_common.indexed_storage import make_IndexedStorage
from PySDM.impl.particle_attributes_factory import ParticleAttributesFactory

from ..dummy_environment import DummyEnvironment
from ..dummy_particulator import DummyParticulator


def make_indexed_storage(backend, iterable, idx=None):
    index = make_Index(backend).from_ndarray(np.array(iterable))
    if idx is not None:
        result = make_IndexedStorage(backend).indexed(idx, index)
    else:
        result = index
    return result


# pylint: disable=protected-access
class TestParticleAttributes:
    @staticmethod
    @pytest.mark.parametrize(
        "water_mass, multiplicity",
        [
            pytest.param(np.array([1.0, 1, 1, 1]), np.array([1, 1, 1, 1])),
            pytest.param(np.array([1.0, 2, 1, 1]), np.array([2, 0, 2, 0])),
            pytest.param(np.array([1.0, 1, 4]), np.array([5, 0, 0])),
        ],
    )
    def test_housekeeping(backend_class, water_mass, multiplicity):
        # Arrange
        particulator = DummyParticulator(backend_class, n_sd=len(multiplicity))
        attributes = {"multiplicity": multiplicity, "water mass": water_mass}
        particulator.build(attributes, int_caster=np.int64)
        sut = particulator.attributes
        sut.healthy = False

        # Act
        sut.sanitize()
        _ = sut.super_droplet_count

        # Assert
        assert sut.super_droplet_count == (multiplicity != 0).sum()
        assert sut["multiplicity"].to_ndarray().sum() == multiplicity.sum()
        assert (
            sut["water mass"].to_ndarray() * sut["multiplicity"].to_ndarray()
        ).sum() == (water_mass * multiplicity).sum()

    @staticmethod
    @pytest.mark.parametrize(
        "multiplicity, cells, idx, new_idx, cell_start",
        [
            ([1, 1, 1], [2, 0, 1], [2, 0, 1], [1, 2, 0], [0, 1, 2, 3]),
            (
                [0, 1, 0, 1, 1],
                [3, 4, 0, 1, 2],
                [4, 1, 3, 2, 0],
                [3, 4, 1],
                [0, 0, 1, 2, 2, 3],
            ),
            (
                [1, 2, 3, 4, 5, 6, 0],
                [2, 2, 2, 2, 1, 1, 1],
                [0, 1, 2, 3, 4, 5, 6],
                [4, 5, 0, 1, 2, 3],
                [0, 0, 2, 6],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "backend_cls", (CPU, pytest.param(GPU, marks=pytest.mark.xfail(strict=True)))
    )  # TODO #330
    def test_sort_by_cell_id(
        *, backend_cls, multiplicity, cells, idx, new_idx, cell_start
    ):
        # Arrange
        n_sd = len(multiplicity)
        particulator = DummyParticulator(backend_cls, n_sd=n_sd)
        n_cell = max(cells) + 1
        particulator.environment.mesh.n_cell = n_cell
        particulator.build(attributes={"multiplicity": np.ones(n_sd)})
        sut = particulator.attributes
        sut._ParticleAttributes__idx = make_indexed_storage(particulator.backend, idx)
        sut._ParticleAttributes__attributes["multiplicity"].data = make_indexed_storage(
            particulator.backend, multiplicity, sut._ParticleAttributes__idx
        )
        sut._ParticleAttributes__attributes["cell id"].data = make_indexed_storage(
            particulator.backend, cells, sut._ParticleAttributes__idx
        )
        sut._ParticleAttributes__cell_start = make_indexed_storage(
            particulator.backend, [0] * (n_cell + 1)
        )
        sut._ParticleAttributes__n_sd = particulator.n_sd
        sut.healthy = 0 not in multiplicity
        sut._ParticleAttributes__cell_caretaker = (
            particulator.backend.make_cell_caretaker(
                sut._ParticleAttributes__idx.shape,
                sut._ParticleAttributes__idx.dtype,
                len(sut._ParticleAttributes__cell_start),
            )
        )

        # Act
        sut.sanitize()
        sut._ParticleAttributes__sort_by_cell_id()

        # Assert
        np.testing.assert_array_equal(
            np.array(new_idx),
            sut._ParticleAttributes__idx.to_ndarray()[: sut.super_droplet_count],
        )
        np.testing.assert_array_equal(
            np.array(cell_start), sut._ParticleAttributes__cell_start.to_ndarray()
        )

    @staticmethod
    def test_recalculate_cell_id(backend_class):
        # Arrange
        multiplicity = np.ones(1, dtype=np.int64)
        droplet_id = 0
        initial_position = np.array([[0], [0]])
        grid = (1, 1)
        particulator = DummyParticulator(backend_class, n_sd=1)
        particulator.environment = DummyEnvironment(grid=grid)
        cell_id, cell_origin, position_in_cell = particulator.mesh.cellular_attributes(
            initial_position
        )
        cell_origin[0, droplet_id] = 0.1
        cell_origin[1, droplet_id] = 0.2
        cell_id[droplet_id] = -1
        attribute = {
            "multiplicity": multiplicity,
            "cell id": cell_id,
            "cell origin": cell_origin,
            "position in cell": position_in_cell,
        }
        particulator.build(attribute)

        # Act
        particulator.recalculate_cell_id()

        # Assert
        assert particulator.attributes["cell id"][droplet_id] == 0

    @staticmethod
    def test_permutation_global_as_implemented_in_numba():
        n_sd = 8
        u01 = [0.1, 0.4, 0.2, 0.5, 0.9, 0.1, 0.6, 0.3]

        # Arrange
        particulator = DummyParticulator(CPU, n_sd=n_sd)
        sut = ParticleAttributesFactory.empty_particles(particulator, n_sd)
        idx_length = len(sut._ParticleAttributes__idx)
        sut._ParticleAttributes__tmp_idx = make_indexed_storage(
            particulator.backend, [0] * idx_length
        )
        sut._ParticleAttributes__sorted = True
        sut._ParticleAttributes__n_sd = particulator.n_sd
        u01 = make_indexed_storage(particulator.backend, u01)

        # Act
        sut.permutation(u01, local=False)

        # Assert
        expected = np.array([1, 3, 5, 7, 6, 0, 4, 2])
        np.testing.assert_array_equal(sut._ParticleAttributes__idx, expected)
        assert not sut._ParticleAttributes__sorted

    @staticmethod
    def test_permutation_local(backend_class):
        if backend_class is GPU:
            pytest.skip("TODO #358")
        n_sd = 8
        u01 = [0.1, 0.4, 0.2, 0.5, 0.9, 0.1, 0.6, 0.3]
        cell_start = [0, 0, 2, 5, 7, n_sd]

        # Arrange
        particulator = DummyParticulator(backend_class, n_sd=n_sd)
        sut = ParticleAttributesFactory.empty_particles(particulator, n_sd)
        idx_length = len(sut._ParticleAttributes__idx)
        sut._ParticleAttributes__tmp_idx = make_indexed_storage(
            particulator.backend, [0] * idx_length
        )
        sut._ParticleAttributes__cell_start = make_indexed_storage(
            particulator.backend, cell_start
        )
        sut._ParticleAttributes__sorted = True
        sut._ParticleAttributes__n_sd = particulator.n_sd
        u01 = make_indexed_storage(particulator.backend, u01)

        # Act
        sut.permutation(u01, local=True)

        # Assert
        expected = np.array([1, 0, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(sut._ParticleAttributes__idx, expected)
        assert sut._ParticleAttributes__sorted

    @staticmethod
    def test_permutation_global_repeatable(backend_class):
        if backend_class is ThrustRTC:
            pytest.skip("TODO #328")

        n_sd = 800
        u01 = np.random.random(n_sd)

        # Arrange
        particulator = DummyParticulator(backend_class, n_sd=n_sd)
        sut = ParticleAttributesFactory.empty_particles(particulator, n_sd)
        idx_length = len(sut._ParticleAttributes__idx)
        sut._ParticleAttributes__tmp_idx = make_indexed_storage(
            particulator.backend, [0] * idx_length
        )
        sut._ParticleAttributes__sorted = True
        u01 = make_indexed_storage(particulator.backend, u01)

        # Act
        sut.permutation(u01, local=False)
        expected = sut._ParticleAttributes__idx.to_ndarray()
        sut._ParticleAttributes__sorted = True
        sut._ParticleAttributes__idx = make_indexed_storage(
            particulator.backend, range(n_sd)
        )
        sut.permutation(u01, local=False)

        # Assert
        np.testing.assert_array_equal(sut._ParticleAttributes__idx, expected)
        assert not sut._ParticleAttributes__sorted

    @staticmethod
    def test_permutation_local_repeatable(backend_class):
        if backend_class is GPU:
            pytest.skip("TODO #358")
        n_sd = 800
        idx = range(n_sd)
        u01 = np.random.random(n_sd)
        cell_start = [0, 0, 20, 250, 700, n_sd]

        # Arrange
        particulator = DummyParticulator(backend_class, n_sd=n_sd)
        cell_id = []
        particulator.environment.mesh.n_cell = len(cell_start) - 1
        for i in range(particulator.environment.mesh.n_cell):
            cell_id += [i] * (cell_start[i + 1] - cell_start[i])
        assert len(cell_id) == n_sd
        particulator.build(attributes={"multiplicity": np.ones(n_sd)})
        sut = particulator.attributes
        sut._ParticleAttributes__idx = make_indexed_storage(particulator.backend, idx)
        idx_length = len(sut._ParticleAttributes__idx)
        sut._ParticleAttributes__tmp_idx = make_indexed_storage(
            particulator.backend, [0] * idx_length
        )
        sut._ParticleAttributes__attributes["cell id"].data = make_indexed_storage(
            particulator.backend, cell_id
        )
        sut._ParticleAttributes__cell_start = make_indexed_storage(
            particulator.backend, cell_start
        )
        sut._ParticleAttributes__sorted = True
        sut._ParticleAttributes__n_sd = particulator.n_sd
        u01 = make_indexed_storage(particulator.backend, u01)

        # Act
        sut.permutation(u01, local=True)
        expected = sut._ParticleAttributes__idx.to_ndarray()
        sut._ParticleAttributes__idx = make_indexed_storage(particulator.backend, idx)
        sut.permutation(u01, local=True)

        # Assert
        np.testing.assert_array_equal(
            sut._ParticleAttributes__idx.to_ndarray(), expected
        )
        assert sut._ParticleAttributes__sorted

        sut._ParticleAttributes__sort_by_cell_id()
        np.testing.assert_array_equal(
            sut._ParticleAttributes__idx.to_ndarray(), expected
        )
