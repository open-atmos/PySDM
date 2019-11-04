"""
Created at 18.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import pytest


@pytest.fixture(params=[
    1
])
def halo(request):
    return request.param
