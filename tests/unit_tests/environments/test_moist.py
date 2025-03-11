"""check for ice-phase-related commons in environment codes"""

import pytest
import numpy as np
from PySDM.environments import Parcel
from PySDM.physics import si, constants_defaults
from PySDM import Builder
from PySDM.backends import GPU

COMMON_PARCEL_CTOR_ARGS = {
    "mixed_phase": True,
    "dt": np.nan,
    "mass_of_dry_air": np.nan,
    "w": np.nan,
}
T0 = constants_defaults.T0


@pytest.mark.parametrize(
    "env, check",
    (
        (
            Parcel(
                p0=1000 * si.hPa,
                initial_water_vapour_mixing_ratio=20 * si.g / si.kg,
                T0=300 * si.K,
                **COMMON_PARCEL_CTOR_ARGS,
            ),
            f"_['T']>{T0} and _['RH']<1 and _['RH_ice']<1",
        ),
        (
            Parcel(
                p0=1000 * si.hPa,
                initial_water_vapour_mixing_ratio=25 * si.g / si.kg,
                T0=300 * si.K,
                **COMMON_PARCEL_CTOR_ARGS,
            ),
            f"_['T']>{T0} and _['RH']>1 and _['RH_ice']<1",
        ),
        (
            Parcel(
                p0=500 * si.hPa,
                initial_water_vapour_mixing_ratio=0.2 * si.g / si.kg,
                T0=250 * si.K,
                **COMMON_PARCEL_CTOR_ARGS,
            ),
            f"_['T']<{T0} and _['RH']<1 and _['RH_ice']<1",
        ),
        (
            Parcel(
                p0=500 * si.hPa,
                initial_water_vapour_mixing_ratio=1 * si.g / si.kg,
                T0=250 * si.K,
                **COMMON_PARCEL_CTOR_ARGS,
            ),
            f"_['T']<{T0} and _['RH']<1 and _['RH_ice']>1",
        ),
    ),
)
def test_ice_properties(backend_instance, env, check):
    """checks ice-related values in recalculated thermodynamic state make sense"""
    if isinstance(backend_instance, GPU):
        pytest.skip("TODO #1495")

    # arrange
    builder = Builder(n_sd=0, backend=backend_instance, environment=env)
    const = builder.particulator.formulae.constants

    # act
    thermo = {
        key: builder.particulator.environment[key].to_ndarray()[0]
        for key in ("RH", "RH_ice", "a_w_ice", "T")
    }

    # assert
    exec(f"assert {check}", {"_": thermo})  # pylint: disable=exec-used
    if thermo["T"] - const.T0 > 0:
        assert thermo["RH"] > thermo["RH_ice"] > 0
    else:
        assert thermo["RH_ice"] > thermo["RH"] > 0
    np.testing.assert_approx_equal(
        thermo["a_w_ice"] * thermo["RH_ice"], thermo["RH"], significant=10
    )
