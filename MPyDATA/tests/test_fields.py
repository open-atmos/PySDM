"""
Created at 14.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.mpdata.fields.scalar_field import ScalarField
from MPyDATA.mpdata.fields.vector_field import VectorField
import numpy as np
import pytest


class TestScalarField2D:
    def test_fill_halos(self):
        # Arrange
        halo = 2
        data = np.arange(0,9).reshape(3,3)
        sut = ScalarField(data, halo)

        # Act
        sut.fill_halos()

        # Assert
        np.testing.assert_equal(sut.get(), data)

        np.testing.assert_equal(sut.data[:halo,halo:-halo], data[-halo:,:])
        np.testing.assert_equal(sut.data[-halo:,halo:-halo], data[:halo,:])

        np.testing.assert_equal(sut.data[halo:-halo,:halo], data[:,-halo:])
        np.testing.assert_equal(sut.data[halo:-halo,-halo:], data[:,:halo])


class TestVectorField1D:
    @pytest.mark.parametrize("halo", [
        pytest.param(1),
        pytest.param(2),
        pytest.param(3),
    ])
    def test_at(self, halo):
        # Arrange
        idx = 3
        data = np.zeros((10,))
        data[idx] = 44
        sut = VectorField(data=[data], halo=halo)

        # Act
        value = sut.at((halo - 1) + (idx - 0.5))

        # Assert
        assert value == data[idx]


class TestVectorField2D:
    @pytest.mark.parametrize("halo", [
        pytest.param(1),
        pytest.param(2),
        pytest.param(3),
    ])
    def test_at(self, halo):
        # Arrange
        idx = (3, 5)
        data1 = np.arange(0, 10 * 12, 1).reshape(10, 12)
        data2 = np.zeros((9, 13))
        data1[idx] = -1
        sut = VectorField(data=(data1, data2), halo=halo)

        # Act
        value = sut.at((halo - 1) + (idx[0] - 0.5), (halo - 1) + idx[1])

        # Assert
        assert value == data1[idx]

    @pytest.mark.parametrize("halo", [
        pytest.param(1),
        pytest.param(2),
        pytest.param(3),
    ])
    def test_set_axis(self, halo):
        # Arrange
        idx = (0, 0)
        data1 = np.zeros((10, 12))
        data2 = np.zeros((9, 13))
        data2[idx] = 44
        sut = VectorField(data=(data1, data2), halo=halo)

        # Act
        sut.set_axis(1)
        value = sut.at(halo - 1 + idx[0] - 0.5, halo - 1 + idx[1])

        # Assert
        assert value == data2[idx]

