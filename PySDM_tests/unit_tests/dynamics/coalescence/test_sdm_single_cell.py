import numpy as np
import pytest

from PySDM.storages.pair_indicator import make_PairIndicator
from PySDM.storages.indexed_storage import make_IndexedStorage
from PySDM.storages.index import make_Index
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend
from PySDM_tests.unit_tests.dynamics.coalescence.__parametrisation__ import backend_fill
from PySDM_tests.unit_tests.dynamics.coalescence.__parametrisation__ import get_dummy_core_and_sdm
# noinspection PyUnresolvedReferences
from PySDM_tests.unit_tests.dynamics.coalescence.__parametrisation__ import v_2, T_2, n_2


class TestSDMSingleCell:

    @staticmethod
    def test_single_collision(backend, v_2, T_2, n_2):
        # Arrange
        const = 1.
        core, sut = get_dummy_core_and_sdm(backend, len(n_2))
        sut.compute_gamma = lambda prob, rand, is_first_in_pair: backend_fill(prob, 1)
        attributes = {'n': n_2, 'volume': v_2, 'heat': const*T_2*v_2, 'temperature': T_2}
        core.build(attributes)

        # Act
        sut()

        # Assert
        particles = core.particles
        a = particles['n'].to_ndarray()
        b = particles['volume'].to_ndarray()
        c = particles['temperature'].to_ndarray()
        np.testing.assert_approx_equal(
            const * np.sum(particles['n'].to_ndarray() * particles['volume'].to_ndarray() * particles['temperature'].to_ndarray()),
            const * np.sum(n_2 * T_2 * v_2),
            significant=7
        )
        new_T = np.sum(T_2 * v_2) / np.sum(v_2)
        assert np.isin(round(new_T, 7), np.round(particles['temperature'].to_ndarray().astype(float), 7))

        assert np.sum(particles['n'].to_ndarray() * particles['volume'].to_ndarray()) == np.sum(n_2 * v_2)
        assert np.sum(core.particles['n'].to_ndarray()) == np.sum(n_2) - np.amin(n_2)
        if np.amin(n_2) > 0:
            assert np.amax(core.particles['volume'].to_ndarray()) == np.sum(v_2)
        assert np.amax(core.particles['n'].to_ndarray()) == max(np.amax(n_2) - np.amin(n_2), np.amin(n_2))

    @staticmethod
    @pytest.mark.parametrize("n_in, n_out", [
        pytest.param(1, np.array([1, 0])),
        pytest.param(2, np.array([1, 1])),
        pytest.param(3, np.array([2, 1])),
    ])
    def test_single_collision_same_n(backend, n_in, n_out):
        # Arrange
        core, sut = get_dummy_core_and_sdm(backend, 2)
        sut.compute_gamma = lambda prob, rand, is_first_in_pair: backend_fill(prob, 1)
        attributes = {'n': np.full(2, n_in), 'volume': np.full(2, 1.)}
        core.build(attributes)

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(sorted(core.particles['n'].to_ndarray(raw=True)), sorted(n_out))

    @staticmethod
    @pytest.mark.parametrize("p", [
        pytest.param(2),
        pytest.param(4),
        pytest.param(5),
        pytest.param(7),
    ])
    def test_multi_collision(backend, v_2, n_2, p):
        # Arrange
        core, sut = get_dummy_core_and_sdm(backend, len(n_2))

        def _compute_gamma(prob, rand, is_first_in_pair):
            from PySDM.dynamics import Coalescence
            backend_fill(prob, p)
            Coalescence.compute_gamma(sut, prob, rand, is_first_in_pair)

        sut.compute_gamma = _compute_gamma

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
        core, sut = get_dummy_core_and_sdm(backend, len(n))

        def _compute_gamma(prob, rand, is_first_in_pair):
            from PySDM.dynamics import Coalescence
            backend_fill(prob, p, odd_zeros=True)
            Coalescence.compute_gamma(sut, prob, rand, is_first_in_pair)
        sut.compute_gamma = _compute_gamma
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

        core, sut = get_dummy_core_and_sdm(backend, n_sd)

        sut.compute_gamma = lambda prob, rand, is_first_in_pair: backend_fill(
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
        np.testing.assert_approx_equal(actual=actual, desired=desired, significant=8)

    @staticmethod
    def test_compute_gamma(backend):
        # Arrange
        n = 87
        prob = np.linspace(0, 3, n, endpoint=True)
        rand = np.linspace(0, 1, n, endpoint=False)

        expected = lambda p, r: p // 1 + (r < p - p // 1)
        n_sd = 2
        for p in prob:
            for r in rand:
                # Act
                prob_arr = backend.Storage.from_ndarray(np.full((n_sd//2,), p))
                rand_arr = backend.Storage.from_ndarray(np.full((n_sd//2,), r))
                idx = make_Index(backend).from_ndarray(np.arange(n_sd))
                mult = make_IndexedStorage(backend).from_ndarray(idx, np.asarray([expected(p, r), 1]).astype(backend.Storage.INT))
                _ = backend.Storage.from_ndarray(np.zeros(n_sd//2))
                cell_id = backend.Storage.from_ndarray(np.zeros(n_sd, dtype=backend.Storage.INT))

                indicator = make_PairIndicator(backend)(n_sd)
                indicator.indicator[0] = 1
                indicator.indicator[1] = 0

                backend.compute_gamma(prob_arr, rand_arr, mult,
                                      cell_id=cell_id, is_first_in_pair=indicator,
                                      collision_rate=_, collision_rate_deficit=_)

                # Assert
                assert expected(p, r) == prob_arr.to_ndarray()[0]

    @staticmethod
    @pytest.mark.parametrize("optimized_random", (pytest.param(True, id='optimized'), pytest.param(False, id='non-optimized')))
    @pytest.mark.parametrize("adaptive", (pytest.param(True, id='adaptive_dt'), pytest.param(False, id='const_dt')))
    def test_rnd_reuse(backend, optimized_random, adaptive):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO #330

        # Arrange
        n_sd = 256
        n = np.random.randint(1, 64, size=n_sd)
        v = np.random.uniform(size=n_sd)
        n_substeps = 5

        particles, sut = get_dummy_core_and_sdm(backend, n_sd, optimized_random=optimized_random, substeps=n_substeps)
        attributes = {'n': n, 'volume': v}
        particles.build(attributes)

        class CountingRandom(backend.Random):
            calls = 0

            def __call__(self, storage):
                CountingRandom.calls += 1
                super(CountingRandom, self).__call__(storage)

        sut.rnd_opt.rnd = CountingRandom(n_sd, seed=44)
        sut.stats_n_substep[:] = n_substeps
        sut.adaptive = adaptive

        # Act
        sut()

        # Assert
        if sut.rnd_opt.optimized_random:
            assert CountingRandom.calls == 2
        else:
            if adaptive:
                assert 2 <= CountingRandom.calls <= 2 * n_substeps
            else:
                assert CountingRandom.calls == 2 * n_substeps
