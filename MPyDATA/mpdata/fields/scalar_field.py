"""
Created at 02.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.mpdata.fields._scalar_field_1d import ScalarField1D
from MPyDATA.mpdata.fields._scalar_field_2d import ScalarField2D


def ScalarField(data, halo):
    dimension = len(data.shape)

    if dimension == 1:
        return ScalarField1D(data, halo)
    if dimension == 2:
        return ScalarField2D(data, halo)
    if dimension == 3:
        raise NotImplementedError()
    raise ValueError()


def clone(scalar_field):
    data = scalar_field.get()
    return ScalarField(data=data, halo=scalar_field.halo)