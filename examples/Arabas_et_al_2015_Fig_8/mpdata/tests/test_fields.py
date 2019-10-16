"""
Created at 14.10.2019

@author: Piotr Bartman
@author: Micheal Olesik
@author: Sylwester Arabas
"""

from examples.Arabas_et_al_2015_Fig_8.mpdata.fields import VectorField, ScalarField
import numpy as np
import pytest


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
        print()
        print(data1, sut.data)
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

