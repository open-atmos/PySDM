import numpy as np
import pytest

from PySDM.physics import si
import PySDM.products as PySDM_products
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM import Formulae
from PySDM.environments import Parcel



volumes = [10., -10.]
test_volumes = [ (v1, v2) for v1 in volumes for v2 in volumes ]

@pytest.mark.parametrize("particle_volume", test_volumes  )
def test_mixed_phase_moments( particle_volume ):

    particle_volume = np.asarray( particle_volume )
    n_sd = len(particle_volume)
    multiplicty = 1
    
    builder = Builder(
        n_sd=n_sd,
        environment=Parcel(dt=np.nan,
                           mass_of_dry_air=1000 * si.kilogram,
                           p0=850 * si.hectopascals,
                           T0=265 * si.kelvin,
                           initial_water_vapour_mixing_ratio=0.,
                           w=0.,
                           mixed_phase = True,
                           ),
        backend=CPU(
            formulae=Formulae(
                particle_shape_and_density="MixedPhaseSpheres",
            )
        ),
        )

    products = [
        PySDM_products.WaterMixingRatio(
            name="water",radius_range=(0, np.inf)),
        PySDM_products.WaterMixingRatio(
            name="ice",radius_range=(-np.inf, 0)),
        PySDM_products.WaterMixingRatio(
            name="total",radius_range=(-np.inf, np.inf)),
        PySDM_products.ParticleConcentration(
            name='n_water',radius_range=(0, np.inf)),
        PySDM_products.ParticleConcentration(
            name='n_ice',radius_range=(-np.inf,0)),
        PySDM_products.ParticleConcentration(
            name='n_total',radius_range=(-np.inf,np.inf)),
        PySDM_products.MeanRadius(
            name='r_water',radius_range=(0,np.inf)),
        PySDM_products.MeanRadius(
            name='r_ice',radius_range=(-np.inf,0)),
        PySDM_products.MeanRadius(
            name='r_all',radius_range=(-np.inf,np.inf)),
        ]

    particulator = builder.build(
        attributes={
            "multiplicity": np.full(shape=(n_sd,), fill_value=multiplicty),
            "volume": particle_volume,
        },
        products = products
    )

    lwc = particulator.products["water"].get()[0]
    iwc = particulator.products["ice"].get()[0]
    twc = particulator.products["total"].get()[0]

    n_w = particulator.products["n_water"].get()[0]
    n_i = particulator.products["n_ice"].get()[0]
    n_t = particulator.products["n_total"].get()[0]

    r_w = particulator.products["r_water"].get()[0]
    r_i = particulator.products["r_ice"].get()[0]
    r_t = particulator.products["r_all"].get()[0]
    
    
    print( lwc, iwc, twc )
    print( n_w, n_i, n_t )
    print( r_w, r_i, r_t )


    # act
    assert( lwc + iwc == twc )
    assert( n_w + n_i == n_t )

    assert( np.isfinite([ lwc, iwc, twc ]).all() )
    assert( np.isfinite([ n_w, n_i, n_t ]).all() )
    assert( np.isfinite([ r_w, r_i, r_t ]).all() )

    # These should also be true:
    # assert( all( (lwc, iwc, twc) ) >= 0 )
    # assert( lwc + abs(iwc) == twc )

    
