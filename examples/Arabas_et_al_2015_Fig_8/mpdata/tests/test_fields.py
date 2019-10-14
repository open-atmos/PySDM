"""
Created at 14.10.2019

@author: Piotr Bartman
@author: Micheal Olesik
@author: Sylwester Arabas
"""

from examples.Arabas_et_al_2015_Fig_8.mpdata.fields import VectorField, ScalarField
import numpy as np


class TestVectorField1D:
    def test_at(self):
        # Arrange
        idx = 3
        data = np.zeros((10,))
        data[idx] = 44
        sut = VectorField(data=[data], halo=1)

        # Act
        value = sut.at(idx - 0.5)

        # Assert
        assert value == data[idx]


class TestVectorField2D:
    def test_at(self):
        # Arrange
        idx = (3, 5)
        data1 = np.zeros((10, 12))
        data2 = np.zeros((9, 13))
        data1[idx] = 44
        sut = VectorField(data=(data1, data2), halo=1)

        # Act
        value = sut.at(idx[0] - 0.5, idx[1])

        # Assert
        assert value == data1[idx]

    def test_set_axis(self):
        # Arrange
        idx = (0, 0)
        data1 = np.zeros((10, 12))
        data2 = np.zeros((9, 13))
        data2[idx] = 44
        sut = VectorField(data=(data1, data2), halo=1)

        # Act
        sut.set_axis(1)
        value = sut.at(idx[0] - 0.5, idx[1])

        # Assert
        assert value == data2[idx]

