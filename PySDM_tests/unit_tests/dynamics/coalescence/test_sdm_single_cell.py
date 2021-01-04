"""
Created at 06.06.2019
"""

import numpy as np
import pytest

from PySDM.dynamics import Coalescence
from PySDM.environments import Box
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend
from PySDM_tests.unit_tests.dummy_core import DummyCore
from PySDM_tests.unit_tests.dynamics.coalescence.__parametrisation__ import StubKernel, backend_fill
# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.dynamics.coalescence.__parametrisation__ import v_2, T_2, n_2


class TestSDMSingleCell:

    @staticmethod
    def get_dummy_core_and_sdm(backend, n_length):
        core = DummyCore(backend, n_sd=n_length)
        dv = 1
        core.environment = Box(dv=dv, dt=0)
        sdm = Coalescence(StubKernel(core.backend))
        sdm.register(core)
        return core, sdm

    @staticmethod
    def test_single_collision(backend, v_2, T_2, n_2):
        # Arrange
        core, sut = TestSDMSingleCell.get_dummy_core_and_sdm(backend, len(n_2))
        sut.compute_gamma = lambda prob, rand: backend_fill(prob, 1)
        attributes = {'n': n_2, 'volume': v_2, 'temperature': T_2}
        core.build(attributes)

        # Act
        sut()

        # Assert
        particles = core.particles
        assert np.sum(particles['n'].to_ndarray() * particles['volume'].to_ndarray() * particles['temperature'].to_ndarray()) == np.sum(n_2 * T_2 * v_2)
        new_T = np.sum(T_2 * v_2) / np.sum(v_2)
        assert np.isin(round(new_T, 10), np.round(particles['temperature'].to_ndarray(), 10))

        assert np.sum(particles['n'].to_ndarray() * particles['volume'].to_ndarray()) == np.sum(n_2 * v_2)
        assert np.sum(core.particles['n'].to_ndarray()) == np.sum(n_2) - np.amin(n_2)
        if np.amin(n_2) > 0: assert np.amax(core.particles['volume'].to_ndarray()) == np.sum(v_2)
        assert np.amax(core.particles['n'].to_ndarray()) == max(np.amax(n_2) - np.amin(n_2), np.amin(n_2))

    @staticmethod
    @pytest.mark.parametrize("n_in, n_out", [
        pytest.param(1, np.array([1, 0])),
        pytest.param(2, np.array([1, 1])),
        pytest.param(3, np.array([2, 1])),
    ])
    def test_single_collision_same_n(backend, n_in, n_out):
        # Arrange
        core, sut = TestSDMSingleCell.get_dummy_core_and_sdm(backend, 2)
        sut.compute_gamma = lambda prob, rand: backend_fill(prob, 1)
        attributes = {'n': np.full(2, n_in), 'volume': np.full(2, 1.)}
        core.build(attributes)

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(sorted(core.particles['n'].to_ndarray()), sorted(n_out))

    @staticmethod
    @pytest.mark.parametrize("p", [
        pytest.param(2),
        pytest.param(4),
        pytest.param(5),
        pytest.param(7),
    ])
    def test_multi_collision(backend, v_2, n_2, p):
        # Arrange
        core, sut = TestSDMSingleCell.get_dummy_core_and_sdm(backend, len(n_2))
        sut.compute_gamma = lambda prob, rand: backend_fill(prob, p)
        attributes = {'n': n_2, 'volume': v_2}
        core.build(attributes)

        # Act
        sut()

        # Assert
        state = core.particles
        gamma = min(p, max(n_2[0] // n_2[1], n_2[1] // n_2[1]))
        assert np.amin(state['n']) >= 0
        assert np.sum(state['n'].to_ndarray() * state['volume'].to_ndarray()) == np.sum(n_2 * v_2)
        assert np.sum(state['n'].to_ndarray()) == np.sum(n_2) - gamma * np.amin(n_2)
        assert np.amax(state['volume'].to_ndarray()) == gamma * v_2[np.argmax(n_2)] + v_2[np.argmax(n_2) - 1]
        assert np.amax(state['n'].to_ndarray()) == max(np.amax(n_2) - gamma * np.amin(n_2), np.amin(n_2))

    @staticmethod
    @pytest.mark.parametrize("v, n, p", [
        pytest.param(np.array([1., 1, 1]), np.array([1, 1, 1]), 2),
        pytest.param(np.array([1., 1, 1, 1, 1]), np.array([5, 1, 2, 1, 1]), 1),
        pytest.param(np.array([1., 1, 1, 1, 1]), np.array([5, 1, 2, 1, 1]), 6),
    ])
    def test_multi_droplet(backend, v, n, p):
        # Arrange
        core, sut = TestSDMSingleCell.get_dummy_core_and_sdm(backend, len(n))
        sut.compute_gamma = lambda prob, rand: backend_fill(prob, p, True)
        attributes = {'n': n, 'volume': v}
        core.build(attributes)

        # Act
        sut()

        # Assert
        assert np.amin(core.particles['n'].to_ndarray()) >= 0
        assert np.sum(core.particles['n'].to_ndarray() * core.particles['volume'].to_ndarray()) == np.sum(n * v)

    @staticmethod
    def test_multi_step(backend):
        # Arrange
        n_sd = 256
        n = np.random.randint(1, 64, size=n_sd)
        v = np.random.uniform(size=n_sd)

        core, sut = TestSDMSingleCell.get_dummy_core_and_sdm(backend, n_sd)

        sut.compute_gamma = lambda prob, rand: backend_fill(
            prob,
            rand.to_ndarray() > 0.5,
            odd_zeros=True
        )
        attributes = {'n': n, 'volume': v}
        core.build(attributes)

        # Act
        for _ in range(32):
            sut()
            core.particles.sanitize()

        # Assert
        assert np.amin(core.particles['n'].to_ndarray()) >= 0
        actual = np.sum(core.particles['n'].to_ndarray() * core.particles['volume'].to_ndarray())
        desired = np.sum(n * v)
        np.testing.assert_almost_equal(actual=actual, desired=desired)

    @staticmethod
    def test_compute_gamma(backend):
        # Arrange
        n = 87
        prob = np.linspace(0, 3, n, endpoint=True)
        rand = np.linspace(0, 1, n, endpoint=False)

        expected = lambda p, r: p // 1 + (r < p - p // 1)

        for p in prob:
            for r in rand:
                # Act
                prob_arr = backend.Storage.from_ndarray(np.full((1,), p))
                rand_arr = backend.Storage.from_ndarray(np.full((1,), r))
                backend.compute_gamma(prob_arr, rand_arr)

                # Assert
                assert expected(p, r) == prob_arr.to_ndarray()[0]

    @staticmethod
    @pytest.mark.parametrize("optimized_random", (True, False))
    def test_rnd_reuse(backend, optimized_random):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO!!!

        # Arrange
        n_sd = 256
        n = np.random.randint(1, 64, size=n_sd)
        v = np.random.uniform(size=n_sd)

        particles, sut = TestSDMSingleCell.get_dummy_core_and_sdm(backend, n_sd)
        attributes = {'n': n, 'volume': v}
        particles.build(attributes)

        class CountingRandom(backend.Random):
            calls = 0

            def __call__(self, storage):
                CountingRandom.calls += 1
                super(CountingRandom, self).__call__(storage)

        sut.rnd_opt.rnd = CountingRandom(n_sd)
        sut.rnd_opt.optimized_random = optimized_random
        sut.substep_num = 100

        # Act
        sut()

        # Assert
        if sut.rnd_opt.optimized_random:
            assert CountingRandom.calls == 2
        else:
            assert CountingRandom.calls == 2 * sut.substep_num
