"""
Created at 06.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import pytest
import numpy as np


'''
x parametrisation: x_2
'''

__x__ = {'ones_2': pytest.param(np.array([1., 1.])),
         'random_2': pytest.param(np.array([4., 2.]))
         }


@pytest.fixture(params=[
    __x__['ones_2'],
    __x__['random_2']
])
def x_2(request):
    return request.param


'''
n parametrisation: 
'''

__n__ = {'1_1': pytest.param(np.array([1, 1])),
         '5_1': pytest.param(np.array([5, 1])),
         '5_3': pytest.param(np.array([5, 3]))
         }


@pytest.fixture(params=[
    __n__['1_1'],
    __n__['5_1'],
    __n__['5_3']
])
def n_2(request):
    return request.param
