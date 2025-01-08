""" check for ice-phase-related commons in environment codes """

import pytest
import numpy as np
from PySDM.environments import Parcel
from PySDM.physics import si
from PySDM import Builder
from PySDM.backends import GPU


@pytest.mark.parametrize(
    "env",
    (
        Parcel(
            mixed_phase=True,
            dt=np.nan,
            mass_of_dry_air=np.nan,
            p0=1000 * si.hPa,
            initial_water_vapour_mixing_ratio=20 * si.g / si.kg,
            T0=300 * si.K,
            w=np.nan,
        ),
    ),
)
def test_ice_properties(backend_instance, env):
    """checks ice-related values in recalculated thermodynamic state make sense"""
    if isinstance(backend_instance, GPU):
        pytest.skip("TODO #1495")

    # arrange
    builder = Builder(n_sd=0, backend=backend_instance, environment=env)

    # act
    thermo = {
        key: builder.particulator.environment[key].to_ndarray()[0]
        for key in ("RH", "RH_ice", "a_w_ice")
    }

    # assert
    assert 1 > thermo["RH"] > thermo["RH_ice"] > 0
    np.testing.assert_approx_equal(
        thermo["a_w_ice"] * thermo["RH_ice"], thermo["RH"], significant=10
    )
