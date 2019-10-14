"""
Created at 14.10.2019

@author: Piotr Bartman
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
        sut = VectorField(data, 0)

        # Act
        value = sut.at(idx - 0.5)

        # Assert
        assert value == data[idx]


class TestVectorField2D:
    def test_at(self):
        pass