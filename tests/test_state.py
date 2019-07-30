from SDM.state import State
from SDM.discretisations import linear
from SDM.spectra import Lognormal

import numpy as np
import pytest


class TestState:

    @staticmethod
    def check_contiguity(state, attr='n', i_SD=0):
        item = state[attr]
        assert item.flags['C_CONTIGUOUS']
        assert item.flags['F_CONTIGUOUS']

        sd = state.get_SD(i_SD)
        assert not sd.flags['C_CONTIGUOUS']
        assert not sd.flags['F_CONTIGUOUS']

    @pytest.mark.xfail
    def test_get_item_does_not_copy(self):
        # Arrange
        arr = np.ones(10)
        sut = State({'n': arr, 'a': arr, 'b': arr})

        # Act
        item = sut['a']

        # Assert
        assert item.base.__array_interface__['data'] == sut.data.__array_interface__['data']

    @pytest.mark.xfail
    def test_get_sd_does_not_copy(self):
        # Arrange
        arr = np.ones(10)
        sut = State({'n': arr, 'a': arr, 'b': arr})

        # Act
        item = sut.get_SD(5)

        # Assert
        assert item.base.__array_interface__['data'] == sut.data.__array_interface__['data']

    @pytest.mark.xfail
    def test_contiguity(self):
        # Arrange
        arr = np.ones(10)
        sut = State({'n': arr, 'a': arr, 'b': arr})

        # Act & Assert
        self.check_contiguity(sut, attr='a', i_SD=5)

    def test_reindex_works(self):
        pass

    @pytest.mark.xfail
    def test_reindex_maintains_contiguity(self):
        # Arrange
        arr = np.linspace(0, 10)
        sut = State({'n': arr, 'a': arr, 'b': arr})
        idx = range(len(arr) - 1, -1, -1)
        assert len(idx) == sut.data.shape[1]

        # Act
        sut._reindex(idx)

        # Assert
        self.check_contiguity(sut)

    def test_moment(self):
        # Arrange (parameters from Clark 1976)
        n_part = 10000  # 190 # cm-3 # TODO!!!!
        mmean = 2e-6  # 6.0 # um    # TODO: geom mean?
        d = 1.2  # dimensionless -> geom. standard dev

        mmin = 0.01e-6
        mmax = 10e-6
        n_sd = 32

        spectrum = Lognormal(n_part, mmean, d)
        x, n = linear(n_sd, spectrum, (mmin, mmax))
        sut = State(n=n, extensive={'x': x}, intensive={}, segment_num=1)

        #debug.plot(sut)
        true_mean, true_var = spectrum.stats(moments='mv')

        # Act
        discr_zero = sut.moment(0) / n_part
        discr_mean = sut.moment(1) / n_part
        discr_mrsq = sut.moment(2) / n_part

        # Assert
        assert abs(discr_zero - 1) / 1 < 1e-3

        assert abs(discr_mean - true_mean) / true_mean < .01e-1

        true_mrsq = true_var + true_mean**2
        assert abs(discr_mrsq - true_mrsq) / true_mrsq < .05e-1

    @pytest.mark.parametrize("x, n", [
        pytest.param(np.array([1., 1, 1, 1]), np.array([1, 1, 1, 1])),
        pytest.param(np.array([1., 2, 1, 1]), np.array([2, 0, 2, 0])),
        pytest.param(np.array([1., 1, 4]), np.array([5, 0, 0]))
    ])
    def test_housekeeping(self, x, n):
        # Arrange
        sut = State(n=n, extensive={'x': x}, intensive={}, segment_num=1)

        # Act
        sut.housekeeping()

        # Assert
        assert sut['x'].shape == sut['n'].shape
        assert sut.SD_num == (n != 0).sum()
        assert sut['n'].sum() == n.sum()
        assert (sut['x'] * sut['n']).sum() == (x * n).sum()
