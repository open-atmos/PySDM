"""
Created at 06.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.colliders import SDM
from SDM.state import State
import numpy as np
import pytest


class StubKernel:
    def __init__(self, returned_value=-1):
        self.returned_value = returned_value

    def __call__(self, backend, output, state):
        output.fill(self.returned_value)


class TestSDM:

    @pytest.mark.parametrize("x, n", [
        pytest.param(np.array([1., 1.]), np.array([1, 1])),
        pytest.param(np.array([1., 1.]), np.array([5, 1])),
        pytest.param(np.array([1., 1.]), np.array([5, 3])),
        pytest.param(np.array([4., 2.]), np.array([1, 1])),
    ])
    def test_single_collision(self, x, n):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=1, n_sd=len(n))
        sut.compute_gamma = lambda backend, prob, rand: prob.fill(1)
        state = State(n=n, extensive={'x': x}, intensive={}, segment_num=1)

        # Act
        sut(state)

        # Assert
        assert np.sum(state['n'] * state['x']) == np.sum(n * x)
        assert np.sum(state['n']) == np.sum(n) - np.amin(n)
        if np.amin(n) > 0: assert np.amax(state['x']) == np.sum(x)
        assert np.amax(state['n']) == max(np.amax(n) - np.amin(n), np.amin(n))

    @pytest.mark.parametrize("n_in, n_out", [
        pytest.param(1, np.array([1, 0])),
        pytest.param(2, np.array([1, 1])),
        pytest.param(3, np.array([2, 1])),
    ])
    def test_single_collision_same_n(self, n_in, n_out):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=1, n_sd=2)
        sut.compute_gamma = lambda backend, prob, rand: prob.fill(1)
        state = State(n=np.full(2, n_in), extensive={'x': np.full(2, 1.)}, intensive={}, segment_num=1)

        # Act
        sut(state)

        # Assert
        np.testing.assert_array_equal(sorted(state._n), sorted(n_out))

    @pytest.mark.parametrize("x, n, p", [
        pytest.param(np.array([1., 1]), np.array([1, 1]), 2),
        pytest.param(np.array([1., 1]), np.array([5, 1]), 4),
        pytest.param(np.array([1., 1]), np.array([5, 3]), 5),
        pytest.param(np.array([4., 2]), np.array([1, 1]), 7),
    ])
    def test_multi_collision(self, x, n, p):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=1, n_sd=len(n))
        sut.compute_gamma = lambda backend, prob, rand: prob.fill(p)
        state = State(n=n, extensive={'x': x}, intensive={}, segment_num=1)

        # Act
        sut(state)

        # Assert
        gamma = min(p, max(n[0] // n[1], n[1] // n[1]))
        assert np.amin(state['n']) >= 0
        assert np.sum(state['n'] * state['x']) == np.sum(n * x)
        assert np.sum(state['n']) == np.sum(n) - gamma * np.amin(n)
        assert np.amax(state['x']) == gamma * x[np.argmax(n)] + x[np.argmax(n) - 1]
        assert np.amax(state['n']) == max(np.amax(n) - gamma * np.amin(n), np.amin(n))

    @pytest.mark.parametrize("x, n, p", [
        pytest.param(np.array([1., 1, 1]), np.array([1, 1, 1]), 2),
        pytest.param(np.array([1., 1, 1, 1, 1]), np.array([5, 1, 2, 1, 1]), 1),
        pytest.param(np.array([1., 1, 1, 1, 1]), np.array([5, 1, 2, 1, 1]), 6),
    ])
    def test_multi_droplet(self, x, n, p):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=1, n_sd=len(n))
        sut.compute_gamma = lambda backend, prob, rand: prob.fill(p)
        state = State(n=n, extensive={'x': x}, intensive={}, segment_num=1)

        # Act
        sut(state)

        # Assert
        assert np.amin(state['n']) >= 0
        assert np.sum(state['n'] * state['x']) == np.sum(n * x)

    # TODO integration test?
    def test_multi_step(self):
        # Arrange
        n = np.random.randint(1, 64, size=256)
        x = np.random.uniform(size=256)

        sut = SDM(StubKernel(), dt=0, dv=1, n_sd=len(n))

        sut.compute_gamma = lambda backend, prob, rand: np.copyto(prob, rand > 0.5)
        state = State(n=n, extensive={'x': x}, intensive={}, segment_num=1)

        # Act
        for _ in range(32):
            sut(state)

        # Assert
        assert np.amin(state['n']) >= 0
        actual = np.sum(state['n'] * state['x'])
        desired = np.sum(n * x)
        np.testing.assert_almost_equal(actual=actual, desired=desired)

    # TODO test_compute_norm_factor
    @pytest.mark.xfail
    def test_probability(self):
        # Arrange
        kernel_value = 44
        dt = 666
        dv = 9
        n_sd = 64
        sut = SDM(StubKernel(kernel_value), dt, dv, n_sd)

        # Act
        actual = sut.probability(1, 1, 0, 0, n_sd)  # TODO dependency state []

        # Assert
        desired = dt/dv * kernel_value * n_sd * (n_sd - 1) / 2 / (n_sd//2)
        assert actual == desired


    def test_compute_gamma(self):
        # Arrange
        n = 87
        prob = np.linspace(0, 3, n, endpoint=True)
        rand = np.linspace(0, 1, n, endpoint=False)

        from SDM.backends.default import Default
        backend = Default

        expected = lambda p,r: p // 1 + (r < p - p // 1)

        for p in prob:
            for r in rand:
                # Act
                prob_arr = backend.from_ndarray(np.full((1,), p))
                rand_arr = backend.from_ndarray(np.full((1,), r))
                SDM.compute_gamma(backend, prob_arr, rand_arr)

                # Assert
                assert expected(p, r) == prob_arr[0]
