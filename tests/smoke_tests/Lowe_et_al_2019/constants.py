import pytest
from PySDM.physics.surface_tension import compressed_film_Ovadnevaite
from PySDM.physics import si
import numpy as np


@pytest.fixture()
def constants():
    compressed_film_Ovadnevaite.sgm_org = 40 * si.mN / si.m
    compressed_film_Ovadnevaite.delta_min = 0.1 * si.nm  # TODO #604 0.2 in the paper, but 0.1 matches the paper plots

    yield

    compressed_film_Ovadnevaite.sgm_org = np.nan
    compressed_film_Ovadnevaite.delta_min = np.nan
