import numpy as np
import pytest

from PySDM.storages.index import make_Index
from PySDM.storages.indexed_storage import make_IndexedStorage
from PySDM.state.particles_factory import ParticlesFactory
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend
from PySDM_tests.unit_tests.dummy_core import DummyCore
from PySDM_tests.unit_tests.dummy_environment import DummyEnvironment
from PySDM.backends import CPU, GPU


class TestParticles:

    @staticmethod
    def make_indexed_storage(backend, iterable, idx=None):
        index = make_Index(backend).from_ndarray(np.array(iterable))
        if idx is not None:
            result = make_IndexedStorage(backend).indexed(idx, index)
        else:
            result = index
        return result

    @staticmethod
    @pytest.mark.parametrize("volume, n", [
        pytest.param(np.array([1., 1, 1, 1]), np.array([1, 1, 1, 1])),
        pytest.param(np.array([1., 2, 1, 1]), np.array([2, 0, 2, 0])),
        pytest.param(np.array([1., 1, 4]), np.array([5, 0, 0]))
    ])
    def test_housekeeping(backend, volume, n):
        # Arrange
        core = DummyCore(backend, n_sd=len(n))
        attributes = {'n': n, 'volume': volume}
        core.build(attributes, int_caster=np.int64)
        sut = core.particles
        sut.healthy = False

        # Act
        sut.sanitize()
        _ = sut.SD_num

        # Assert
        assert sut.SD_num == (n != 0).sum()
        assert sut['n'].to_ndarray().sum() == n.sum()
        assert (sut['volume'].to_ndarray() * sut['n'].to_ndarray()).sum() == (volume * n).sum()

    @staticmethod
    @pytest.mark.parametrize('n, cells, n_sd, idx, new_idx, cell_start', [
        ([1, 1, 1], [2, 0, 1], 3, [2, 0, 1], [1, 2, 0], [0, 1, 2, 3]),
        ([0, 1, 0, 1, 1], [3, 4, 0, 1, 2], 3, [4, 1, 3, 2, 0], [3, 4, 1], [0, 0, 1, 2, 2, 3]),
        ([1, 2, 3, 4, 5, 6, 0], [2, 2, 2, 2, 1, 1, 1], 6, [0, 1, 2, 3, 4, 5, 6], [4, 5, 0, 1, 2, 3], [0, 0, 2, 6])
    ])
    def test_sort_by_cell_id(backend, n, cells, n_sd, idx, new_idx, cell_start):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO #330

        # Arrange
        core = DummyCore(backend, n_sd=n_sd)
        n_cell = max(cells) + 1
        core.environment.mesh.n_cell = n_cell
        core.build(attributes={'n': np.ones(n_sd)})
        sut = core.particles
        sut._Particles__idx = TestParticles.make_indexed_storage(backend, idx)
        sut.attributes['n'].data = TestParticles.make_indexed_storage(backend, n, sut._Particles__idx)
        sut.attributes['cell id'].data = TestParticles.make_indexed_storage(backend, cells, sut._Particles__idx)
        sut._Particles__cell_start = TestParticles.make_indexed_storage(backend, [0] * (n_cell + 1))
        sut._Particles__n_sd = core.n_sd
        sut.healthy = 0 not in n
        sut._Particles__cell_caretaker = backend.make_cell_caretaker(sut._Particles__idx, sut._Particles__cell_start)

        # Act
        sut.sanitize()
        sut._Particles__sort_by_cell_id()

        # Assert
        np.testing.assert_array_equal(np.array(new_idx), sut._Particles__idx.to_ndarray()[:sut.SD_num])
        np.testing.assert_array_equal(np.array(cell_start), sut._Particles__cell_start.to_ndarray())

    @staticmethod
    def test_recalculate_cell_id(backend):
        # Arrange
        n = np.ones(1, dtype=np.int64)
        droplet_id = 0
        initial_position = np.array([[0], [0]])
        grid = (1, 1)
        core = DummyCore(backend, n_sd=1)
        core.environment = DummyEnvironment(grid=grid)
        cell_id, cell_origin, position_in_cell = core.mesh.cellular_attributes(initial_position)
        cell_origin[0, droplet_id] = .1
        cell_origin[1, droplet_id] = .2
        cell_id[droplet_id] = -1
        attribute = {'n': n, 'cell id': cell_id, 'cell origin': cell_origin, 'position in cell': position_in_cell}
        core.build(attribute)
        sut = core.particles

        # Act
        sut.recalculate_cell_id()

        # Assert
        assert sut['cell id'][droplet_id] == 0

    @staticmethod
    def test_permutation_global_as_implemented_in_Numba():
        n_sd = 8
        u01 = [.1, .4, .2, .5, .9, .1, .6, .3]

        # Arrange
        core = DummyCore(CPU, n_sd=n_sd)
        sut = ParticlesFactory.empty_particles(core, n_sd)
        idx_length = len(sut._Particles__idx)
        sut._Particles__tmp_idx = TestParticles.make_indexed_storage(CPU, [0] * idx_length)
        sut._Particles__sorted = True
        sut._Particles__n_sd = core.n_sd
        u01 = TestParticles.make_indexed_storage(CPU, u01)

        # Act
        sut.permutation(u01, local=False)

        # Assert
        expected = np.array([1, 3, 5, 7, 6, 0, 4, 2])
        np.testing.assert_array_equal(sut._Particles__idx, expected)
        assert not sut._Particles__sorted

    @staticmethod
    def test_permutation_local(backend):
        if backend==GPU:  # TODO #358
            return
        n_sd = 8
        u01 = [.1, .4, .2, .5, .9, .1, .6, .3]
        cell_start = [0, 0, 2, 5, 7, n_sd]

        # Arrange
        core = DummyCore(backend, n_sd=n_sd)
        sut = ParticlesFactory.empty_particles(core, n_sd)
        idx_length = len(sut._Particles__idx)
        sut._Particles__tmp_idx = TestParticles.make_indexed_storage(backend, [0] * idx_length)
        sut._Particles__cell_start = TestParticles.make_indexed_storage(backend, cell_start)
        sut._Particles__sorted = True
        sut._Particles__n_sd = core.n_sd
        u01 = TestParticles.make_indexed_storage(backend, u01)

        # Act
        sut.permutation(u01, local=True)

        # Assert
        expected = np.array([1, 0, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(sut._Particles__idx, expected)
        assert sut._Particles__sorted

    @staticmethod
    def test_permutation_global_repeatable(backend):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO #328

        n_sd = 800
        u01 = np.random.random(n_sd)

        # Arrange
        core = DummyCore(backend, n_sd=n_sd)
        sut = ParticlesFactory.empty_particles(core, n_sd)
        idx_length = len(sut._Particles__idx)
        sut._Particles__tmp_idx = TestParticles.make_indexed_storage(backend, [0] * idx_length)
        sut._Particles__sorted = True
        #sut._Particles__n_sd = core.n_sd
        u01 = TestParticles.make_indexed_storage(backend, u01)

        # Act
        sut.permutation(u01, local=False)
        expected = sut._Particles__idx.to_ndarray()
        sut._Particles__sorted = True
        sut._Particles__idx = TestParticles.make_indexed_storage(backend, range(n_sd))
        sut.permutation(u01, local=False)

        # Assert
        np.testing.assert_array_equal(sut._Particles__idx, expected)
        assert not sut._Particles__sorted

    @staticmethod
    def test_permutation_local_repeatable(backend):
        if backend==GPU:  # TODO #358
            return
        n_sd = 800
        idx = range(n_sd)
        u01 = np.random.random(n_sd)
        cell_start = [0, 0, 20, 250, 700, n_sd]

        # Arrange
        core = DummyCore(backend, n_sd=n_sd)
        cell_id = []
        core.environment.mesh.n_cell = len(cell_start) - 1
        for i in range(core.environment.mesh.n_cell):
            cell_id += [i] * (cell_start[i + 1] - cell_start[i])
        assert len(cell_id) == n_sd
        core.build(attributes={'n': np.ones(n_sd)})
        sut = core.particles
        sut._Particles__idx = TestParticles.make_indexed_storage(backend, idx)
        idx_length = len(sut._Particles__idx)
        sut._Particles__tmp_idx = TestParticles.make_indexed_storage(backend, [0] * idx_length)
        sut.attributes['cell id'].data = TestParticles.make_indexed_storage(backend, cell_id)
        sut._Particles__cell_start = TestParticles.make_indexed_storage(backend, cell_start)
        sut._Particles__sorted = True
        sut._Particles__n_sd = core.n_sd
        u01 = TestParticles.make_indexed_storage(backend, u01)

        # Act
        sut.permutation(u01, local=True)
        expected = sut._Particles__idx.to_ndarray()
        sut._Particles__idx = TestParticles.make_indexed_storage(backend, idx)
        sut.permutation(u01, local=True)

        # Assert
        np.testing.assert_array_equal(sut._Particles__idx.to_ndarray(), expected)
        assert sut._Particles__sorted

        sut._Particles__sort_by_cell_id()
        np.testing.assert_array_equal(sut._Particles__idx.to_ndarray(), expected)
