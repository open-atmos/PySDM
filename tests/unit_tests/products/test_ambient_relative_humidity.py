# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder, products
from PySDM.backends import CPU, GPU
from PySDM.environments import Parcel
from PySDM.physics import si


@pytest.mark.parametrize(
    "backend_class", (CPU, pytest.param(GPU, marks=pytest.mark.xfail(strict=True)))
)
def test_ambient_relative_humidity(backend_class):
    # arrange
    n_sd = 1

    env = Parcel(
        dt=np.nan,
        mixed_phase=True,
        mass_of_dry_air=np.nan,
        p0=1000 * si.hPa,
        initial_water_vapour_mixing_ratio=1 * si.g / si.kg,
        T0=260 * si.K,
        w=np.nan,
    )
    builder = Builder(n_sd, backend=backend_class(), environment=env)
    attributes = {"multiplicity": np.ones(n_sd), "volume": np.ones(n_sd)}
    particulator = builder.build(
        attributes=attributes,
        products=(
            products.AmbientRelativeHumidity(name="RHw", var="RH"),
            products.AmbientRelativeHumidity(name="RHi", var="RH", ice=True),
        ),
    )

    # act
    values = {}
    for name, product in particulator.products.items():
        values[name] = product.get()[0]

    # assert
    assert values["RHw"] < values["RHi"]
