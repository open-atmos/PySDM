"""
Created at 03.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest


@pytest.fixture(params=[
    pytest.param((1,)),
    pytest.param((8,)),
    pytest.param((1, 1)),
    pytest.param((1, 7)),
    pytest.param((8, 1)),
    pytest.param((8, 7)),
    pytest.param((8, 9))
])
def shape_full(request):
    return request.param


@pytest.fixture(params=[
    pytest.param((1, 1)),
    pytest.param((1, 7)),
])
def shape_2D1D(request):
    return request.param


@pytest.fixture(params=[
    pytest.param((1,)),
    pytest.param((8,)),
])
def shape_1D(request):
    return request.param

@pytest.fixture(params=[
    pytest.param((1, 1)),
    pytest.param((1, 7)),
    pytest.param((8, 1)),
    pytest.param((8, 7)),
    pytest.param((8, 9))
])
def shape_2D(request):
    return request.param


@pytest.fixture(params=[
    pytest.param(float),
    pytest.param(int),
    pytest.param(bytes)
])
def dtype_full(request):
    return request.param


@pytest.fixture(params=[
    pytest.param(float),
    pytest.param(int)
])
def dtype(request):
    return request.param


@pytest.fixture(params=[
    pytest.param('zero'),
    pytest.param('middle'),
    pytest.param('full')
])
def length(request):
    return request.param
