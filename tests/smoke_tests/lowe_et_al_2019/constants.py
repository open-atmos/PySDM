# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
import numpy as np
from PySDM.physics.surface_tension import compressed_film_ovadnevaite
from PySDM.physics import si


@pytest.fixture()
def constants():
    compressed_film_ovadnevaite.sgm_org = 40 * si.mN / si.m
    # TODO #604 0.2 in the paper, but 0.1 matches the paper plots
    compressed_film_ovadnevaite.delta_min = 0.1 * si.nm

    yield

    compressed_film_ovadnevaite.sgm_org = np.nan
    compressed_film_ovadnevaite.delta_min = np.nan
