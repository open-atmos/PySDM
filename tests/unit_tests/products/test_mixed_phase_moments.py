# pylint: disable=missing-module-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.physics import si
import PySDM.products as PySDM_products
from PySDM.builder import Builder
from PySDM import Formulae
from PySDM.environments import Box


MASSES = (10.0 * si.ug, -10.0 * si.ug)


@pytest.mark.parametrize(
    "particle_mass", [np.asarray((v1, v2)) for v1 in MASSES for v2 in MASSES]
)
def test_mixed_phase_moments(particle_mass, backend_class):
    # arrange
    particulator = Builder(
        n_sd=len(particle_mass),
        environment=Box(dt=np.nan, dv=1 * si.m**3),
        backend=backend_class(
            formulae=Formulae(
                particle_shape_and_density="MixedPhaseSpheres",
            )
        ),
    ).build(
        attributes={
            "multiplicity": np.full_like(particle_mass, fill_value=1),
            "signed water mass": particle_mass,
        },
        products=(
            PySDM_products.WaterMixingRatio(name="water", radius_range=(0, np.inf)),
            PySDM_products.WaterMixingRatio(name="ice", radius_range=(-np.inf, 0)),
            PySDM_products.WaterMixingRatio(
                name="total", radius_range=(-np.inf, np.inf)
            ),
            PySDM_products.ParticleConcentration(
                name="n_water", radius_range=(0, np.inf)
            ),
            PySDM_products.ParticleConcentration(
                name="n_ice", radius_range=(-np.inf, 0)
            ),
            PySDM_products.ParticleConcentration(
                name="n_total", radius_range=(-np.inf, np.inf)
            ),
            PySDM_products.MeanRadius(name="r_water", radius_range=(0, np.inf)),
            PySDM_products.MeanRadius(name="r_ice", radius_range=(-np.inf, 0)),
            PySDM_products.MeanRadius(name="r_all", radius_range=(-np.inf, np.inf)),
        ),
    )
    particulator.environment["rhod"] = 1 * si.kg / si.m**3

    # act
    lwc = particulator.products["water"].get()[0]
    iwc = particulator.products["ice"].get()[0]
    twc = particulator.products["total"].get()[0]

    n_w = particulator.products["n_water"].get()[0]
    n_i = particulator.products["n_ice"].get()[0]
    n_t = particulator.products["n_total"].get()[0]

    r_w = particulator.products["r_water"].get()[0]
    r_i = particulator.products["r_ice"].get()[0]
    r_t = particulator.products["r_all"].get()[0]

    # assert
    assert np.isfinite([lwc, iwc, twc]).all()
    assert np.isfinite([n_w, n_i, n_t]).all()
    assert np.isfinite([r_w, r_i, r_t]).all()

    if any(particle_mass > 0):
        assert all(product > 0 for product in (lwc, n_w, r_w))
    if any(particle_mass < 0):
        assert all(product > 0 for product in (iwc, n_i, r_i))
    assert twc > 0
    assert lwc + iwc == twc
    assert n_w + n_i == n_t
