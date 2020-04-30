"""
Created at 06.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.dynamics.coalescence.algorithms import SDM
from PySDM_tests.unit_tests.simulation.state.testable_state_factory import TestableStateFactory
from PySDM_tests.unit_tests.simulation.state.dummy_particles import DummyParticles
from PySDM.backends.default import Default
from PySDM.simulation import Box
import numpy as np
import pytest
from PySDM_tests.unit_tests.simulation.dynamics.coalescence.__parametrisation__ import StubKernel, backend_fill

# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.simulation.dynamics.coalescence.__parametrisation__ import v_2, T_2, n_2

backend = Default


class TestSDMSingleCell:

    @staticmethod
    def get_dummy_particles_and_sdm(n_length):
        particles = DummyParticles(backend, n_sd=n_length, dt=0)
        dv = 1
        particles.set_environment(Box, {'dv': dv})
        sdm = SDM(particles, StubKernel(particles.backend))
        return particles, sdm

    def test_single_collision(self, v_2, T_2, n_2):
        # Arrange
        particles, sut = TestSDMSingleCell.get_dummy_particles_and_sdm(len(n_2))
        sut.compute_gamma = lambda prob, rand: backend_fill(backend, prob, 1)
        particles.state = TestableStateFactory.state_0d(n=n_2, extensive={'volume': v_2},
                                                        intensive={'temperature': T_2},
                                                        particles=particles)

        # Act
        sut()

        # Assert
        state = particles.state
        assert np.sum(state['n'] * state['volume'] * state['temperature']) == np.sum(n_2 * T_2 * v_2)
        new_T = np.sum(T_2 * v_2) / np.sum(v_2)
        assert np.isin(round(new_T, 10), np.round(state['temperature'], 10))

        assert np.sum(particles.state['n'] * particles.state['volume']) == np.sum(n_2 * v_2)
        assert np.sum(particles.state['n']) == np.sum(n_2) - np.amin(n_2)
        if np.amin(n_2) > 0: assert np.amax(particles.state['volume']) == np.sum(v_2)
        assert np.amax(particles.state['n']) == max(np.amax(n_2) - np.amin(n_2), np.amin(n_2))

    @pytest.mark.parametrize("n_in, n_out", [
        pytest.param(1, np.array([1, 0])),
        pytest.param(2, np.array([1, 1])),
        pytest.param(3, np.array([2, 1])),
    ])
    def test_single_collision_same_n(self, n_in, n_out):
        # Arrange
        particles, sut = TestSDMSingleCell.get_dummy_particles_and_sdm(2)
        sut.compute_gamma = lambda prob, rand: backend_fill(backend, prob, 1)
        particles.state = TestableStateFactory.state_0d(n=np.full(2, n_in),
                                                        extensive={'volume': np.full(2, 1.)},
                                                        intensive={},
                                                        particles=particles)

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(sorted(particles.backend.to_ndarray(particles.state.n)), sorted(n_out))

    @pytest.mark.parametrize("p", [
        pytest.param(2),
        pytest.param(4),
        pytest.param(5),
        pytest.param(7),
    ])
    def test_multi_collision(self, v_2, n_2, p):
        # Arrange
        particles, sut = TestSDMSingleCell.get_dummy_particles_and_sdm(len(n_2))
        sut.compute_gamma = lambda prob, rand: backend_fill(backend, prob, p)
        particles.state = TestableStateFactory.state_0d(n=n_2, extensive={'volume': v_2}, intensive={},
                                                        particles=particles)

        # Act
        sut()

        # Assert
        state = particles.state
        gamma = min(p, max(n_2[0] // n_2[1], n_2[1] // n_2[1]))
        assert np.amin(state['n']) >= 0
        assert np.sum(state['n'] * state['volume']) == np.sum(n_2 * v_2)
        assert np.sum(state['n']) == np.sum(n_2) - gamma * np.amin(n_2)
        assert np.amax(state['volume']) == gamma * v_2[np.argmax(n_2)] + v_2[np.argmax(n_2) - 1]
        assert np.amax(state['n']) == max(np.amax(n_2) - gamma * np.amin(n_2), np.amin(n_2))

    @pytest.mark.parametrize("x, n, p", [
        pytest.param(np.array([1., 1, 1]), np.array([1, 1, 1]), 2),
        pytest.param(np.array([1., 1, 1, 1, 1]), np.array([5, 1, 2, 1, 1]), 1),
        pytest.param(np.array([1., 1, 1, 1, 1]), np.array([5, 1, 2, 1, 1]), 6),
    ])
    def test_multi_droplet(self, x, n, p):
        # Arrange
        particles, sut = TestSDMSingleCell.get_dummy_particles_and_sdm(len(n))
        sut.compute_gamma = lambda prob, rand: backend_fill(backend, prob, p, True)
        particles.state = TestableStateFactory.state_0d(n=n, extensive={'volume': x}, intensive={},
                                                        particles=particles)

        # Act
        sut()

        # Assert
        assert np.amin(particles.state['n']) >= 0
        assert np.sum(particles.state['n'] * particles.state['volume']) == np.sum(n * x)

    # TODO integration test?
    def test_multi_step(self):
        # Arrange
        n_sd = 256
        n = np.random.randint(1, 64, size=n_sd)
        x = np.random.uniform(size=n_sd)

        particles, sut = TestSDMSingleCell.get_dummy_particles_and_sdm(n_sd)

        sut.compute_gamma = lambda prob, rand: backend_fill(
            backend,
            prob,
            backend.to_ndarray(rand) > 0.5,
            odd_zeros=True
        )
        particles.state = TestableStateFactory.state_0d(n=n, extensive={'volume': x}, intensive={},
                                                        particles=particles)

        # Act
        for _ in range(32):
            sut()

        # Assert
        assert np.amin(particles.state['n']) >= 0
        actual = np.sum(particles.state['n'] * particles.state['volume'])
        desired = np.sum(n * x)
        np.testing.assert_almost_equal(actual=actual, desired=desired)

    # TODO: move to backend tests
    @staticmethod
    def test_compute_gamma():
        # Arrange
        n = 87
        prob = np.linspace(0, 3, n, endpoint=True)
        rand = np.linspace(0, 1, n, endpoint=False)

        from PySDM.backends.default import Default
        backend = Default()

        expected = lambda p, r: p // 1 + (r < p - p // 1)

        for p in prob:
            for r in rand:
                # Act
                prob_arr = backend.from_ndarray(np.full((1,), p))
                rand_arr = backend.from_ndarray(np.full((1,), r))
                backend.compute_gamma(prob_arr, rand_arr)

                # Assert
                assert expected(p, r) == backend.to_ndarray(prob_arr)[0]

    # TODO test_compute_probability
