""" exemplifies and does basic tests of ventilation logic in condensation/evaporation """

import pytest
import numpy as np

from PySDM import Formulae, Builder
from PySDM.environments import Parcel
from PySDM.formulae import _choices
from PySDM.physics import ventilation, si
from PySDM.dynamics import Condensation, AmbientThermodynamics
from PySDM.products import AmbientRelativeHumidity
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver

INITIAL_DROPLET_MASS = 1 * si.ug
DRY_VOLUME = 1 * si.nm**3


def _make_particulator(backend):
    builder = Builder(
        backend=backend,
        n_sd=1,
        environment=Parcel(
            dt=1 * si.s,
            mass_of_dry_air=1 * si.mg,
            p0=1000 * si.hPa,
            initial_water_vapour_mixing_ratio=6.66 * si.g / si.kg,
            T0=285 * si.K,
            w=1 * si.m / si.s,
        ),
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation())
    return builder.build(
        attributes={
            "multiplicity": np.ones(1),
            "water mass": np.asarray([INITIAL_DROPLET_MASS]),
            "dry volume": np.asarray([DRY_VOLUME]),
            "kappa times dry volume": 0.5 * np.asarray([DRY_VOLUME]),
        },
        products=(AmbientRelativeHumidity(name="RH"),),
    )


@pytest.mark.parametrize(
    "variant", [v for v in _choices(ventilation) if v != ventilation.Neglect.__name__]
)
@pytest.mark.parametrize("scipy_solver", (True, False))
def test_ventilation(backend_class, variant, scipy_solver):
    """tests checking effects of ventilation in a simplistic
    single-[super]droplet adiabatic parcel simulation set up to
    trigger evaporation of a large droplet in subsaturated air"""

    # arrange
    particulators = {
        key: _make_particulator(backend_class(formulae=Formulae(ventilation=key)))
        for key in [variant, ventilation.Neglect.__name__]
    }

    if scipy_solver:
        if backend_class.__name__ != "Numba":
            pytest.skip("SciPy solver works only for Numba backend")
        for particulator in particulators.values():
            scipy_ode_condensation_solver.patch_particulator(particulator)

    # act
    for particulator in particulators.values():
        particulator.run(steps=1)
        assert 0.75 < particulator.products["RH"].get()[0] < 0.8

    # assert
    mass_ratios = {
        key: particulator.attributes["water mass"].to_ndarray() / INITIAL_DROPLET_MASS
        for key, particulator in particulators.items()
    }
    assert 0.95 < mass_ratios[variant] < mass_ratios["Neglect"] < 1
