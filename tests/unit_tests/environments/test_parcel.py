"""test for the parcel environment"""

import pytest
import numpy as np
from PySDM.environments import Parcel
from PySDM.physics import si
from PySDM.builder import Builder


def test_exclussive_for_rh0_and_r0():
    # arrange
    args = {
        "dt": 0.25 * si.s,
        "mass_of_dry_air": 1e3 * si.kg,
        "p0": 1000 * si.hPa,
        "T0": 280 * si.K,
        "w": 0.25 * si.m / si.s,
    }

    # act
    with pytest.raises(AssertionError):
        Parcel(
            initial_water_vapour_mixing_ratio=10, initial_relative_humidity=20, **args
        )


def test_rh_to_mixing_ratio_conversion(backend_instance):
    # arrange
    args = {
        "dt": 0.25 * si.s,
        "mass_of_dry_air": 1e3 * si.kg,
        "p0": 1000 * si.hPa,
        "T0": 280 * si.K,
        "w": 0.25 * si.m / si.s,
    }
    rh = 0

    builder = Builder(
        n_sd=1,
        backend=backend_instance,
        environment=Parcel(initial_relative_humidity=rh, **args),
    )

    particulator = builder.build(
        {"multiplicity": np.ones(1), "signed water mass": np.ones(1)}
    )
    # assert
    assert particulator.environment.initial_water_vapour_mixing_ratio == 0
