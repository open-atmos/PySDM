"""asserts that cached and uncached backend instantiation works as expected"""

import pytest

from PySDM.backends import CPU, GPU, Numba, ThrustRTC


class TestInstanceCache:
    @staticmethod
    @pytest.mark.parametrize("fun", (CPU, GPU))
    def test_cache_hit(fun):
        # arrange
        backend1 = fun()

        # act
        backend2 = fun()

        # assert
        assert id(backend1) == id(backend2)

    @staticmethod
    @pytest.mark.parametrize("fun", (Numba, ThrustRTC))
    def test_uncached_ctors(fun):
        # arrange
        backend1 = fun()

        # act
        backend2 = fun()

        # assert
        assert id(backend1) != id(backend2)
