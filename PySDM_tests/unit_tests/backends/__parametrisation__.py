"""
Created at 03.08.2019
"""

import pytest
from PySDM.backends import ThrustRTC

from PySDM.backends.default import Default

backend = Default()
backends = []
if ThrustRTC.ENABLE:
    backends.append(ThrustRTC())


'''
 number parametrisation: number_float, number_int, number
'''

__number__ = {'f_zero': pytest.param(0.),
              'f_one': pytest.param(1.),
              'f_big_positive': pytest.param(100087.),
              'f_small_positive': pytest.param(.00697),
              'f_big_negative_frac': pytest.param(-100005.5723),
              'f_small_negative': pytest.param(-.0051),
              'i_one': pytest.param(1),
              'i_big_positive': pytest.param(100063),
              'i_big_negative': pytest.param(-200039)
              }


@pytest.fixture(params=[
    __number__['f_zero'],
    __number__['f_one'],
    __number__['f_big_positive'],
    __number__['f_small_positive'],
    __number__['f_big_negative_frac'],
    __number__['f_small_negative']
])
def number_float(request):
    return request.param


@pytest.fixture(params=[
    __number__['i_one'],
    __number__['i_big_positive'],
    __number__['i_big_negative']
])
def number_int(request):
    return request.param


@pytest.fixture(params=[*__number__.values()])
def number(request):
    return request.param


'''
shape parametrisation: shape_full, shape_1d, shape_2d
'''

__shape__ = {'1d_1': pytest.param((1,)),
             '1d': pytest.param((33,)),
             '2d_1': pytest.param((1, 1)),
             '2d_row': pytest.param((1, 7)),
             '2d_col': pytest.param((8, 1)),
             '2d_full_short_row': pytest.param((8, 7)),
             '2d_full_long_row': pytest.param((8, 19))
             }


@pytest.fixture(params=[
    __shape__['1d_1'],
    __shape__['1d'],
    __shape__['2d_1'],
    __shape__['2d_row'],
    __shape__['2d_col'],
    __shape__['2d_full_short_row'],
    __shape__['2d_full_long_row']
])
def shape_full(request):
    return request.param


@pytest.fixture(params=[
    __shape__['1d_1'],
    __shape__['1d']
])
def shape_1d(request):
    return request.param


@pytest.fixture(params=[
    __shape__['2d_1'],
    __shape__['2d_row'],
    __shape__['2d_col'],
    __shape__['2d_full_short_row'],
    __shape__['2d_full_long_row']
])
def shape_2d(request):
    return request.param


'''
dtype parametrisation: dtype_full, dtype
'''

__dtype__ = {'float': pytest.param(float),
             'int': pytest.param(int),
             'unsupported': pytest.param(bytes)
             }


@pytest.fixture(params=[
    __dtype__['float'],
    __dtype__['int'],
    __dtype__['unsupported']
])
def dtype_full(request):
    return request.param


@pytest.fixture(params=[
    __dtype__['float'],
    __dtype__['int']
])
def dtype(request):
    return request.param


@pytest.fixture(params=[
    __dtype__['float'],
    __dtype__['int']
])
def dtype_mixed(request):
    return request.param


'''
length parametrisation: length, natural_length
'''

__length__ = {'zero': pytest.param('zero'),
              'middle': pytest.param('middle'),
              'full': pytest.param('full')
              }


@pytest.fixture(params=[
    __length__['zero'],
    __length__['middle'],
    __length__['full']
])
def length(request):
    return request.param


@pytest.fixture(params=[
    __length__['middle'],
    __length__['full']
])
def natural_length(request):
    return request.param


'''
order parametrisation: order
'''

__order__ = {'asc': pytest.param('asc'),
             'desc': pytest.param('desc'),
             'random': pytest.param('random')
             }


@pytest.fixture(params=[
    __order__['asc'],
    __order__['desc'],
    __order__['random']
])
def order(request):
    return request.param


'''
pairs parametrisation: pairs
'''

__pairs__ = {'none': pytest.param('none'),
             'random': pytest.param('random'),
             'full': pytest.param('full')
             }


@pytest.fixture(params=[
    __pairs__['none'],
    __pairs__['random'],
    __pairs__['full']
])
def pairs(request):
    return request.param
