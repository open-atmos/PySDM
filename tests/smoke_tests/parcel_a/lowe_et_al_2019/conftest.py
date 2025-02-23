# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.physics import si
from PySDM.physics.surface_tension import compressed_film_ovadnevaite


@pytest.fixture(name="constants")
def constants_fixture():
    compressed_film_ovadnevaite.sgm_org = 40 * si.mN / si.m
    # TODO #1247 0.2 in the paper, but 0.1 matches the paper plots
    compressed_film_ovadnevaite.delta_min = 0.1 * si.nm

    yield

    compressed_film_ovadnevaite.sgm_org = np.nan
    compressed_film_ovadnevaite.delta_min = np.nan
