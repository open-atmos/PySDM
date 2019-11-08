"""
Created at 02.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.mpdata.fields._vector_field_1d import VectorField1D
from MPyDATA.mpdata.fields._vector_field_2d import VectorField2D, div_2d


def VectorField(data, halo):
    if len(data) == 1:
        return VectorField1D(data[0], halo)
    if len(data) == 2:
        return VectorField2D(data[0], data[1], halo)
    if len(data) == 3:
        raise NotImplementedError()
    else:
        raise ValueError()


def clone(vector_field, value=np.nan):
    data = [np.full_like(vector_field.get_component(d), value) for d in range(vector_field.dimension)]
    return VectorField(data, halo=vector_field.halo)


def div(vector_field, grid_step: tuple):
    if vector_field.dimension == 2:
        return div_2d(vector_field, grid_step)
    raise NotImplementedError()

