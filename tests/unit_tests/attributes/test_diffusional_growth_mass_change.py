import numpy as np
from PySDM import Builder
from PySDM.dynamics import Condensation, AmbientThermodynamics
from PySDM.physics import si
from ..dynamics.test_vapour_deposition_on_ice import MoistBox


def test_diffusional_growth_mass_change(backend_instance):
    # arrange
    n_sd = 1
    dt = 1 * si.s

    builder = Builder(
        backend=backend_instance, n_sd=n_sd, environment=MoistBox(dt=dt, dv=np.nan)
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation())
    builder.request_attribute("diffusional growth mass change")
    particulator = builder.build(
        attributes={
            "water mass": np.ones(n_sd) * si.ng,
            "multiplicity": np.ones(n_sd),
            "dry volume": (dry_volume := np.ones(n_sd) * si.nm**3),
            "kappa times dry volume": np.ones(n_sd) * dry_volume * (kappa := 0.6),
        }
    )
    particulator.environment["rhod"] = 1 * si.kg / si.m**3
    particulator.environment["thd"] = 300 * si.K
    particulator.environment["water_vapour_mixing_ratio"] = 10 * si.g / si.kg

    # act
    particulator.run(steps=1)

    # assert
    diffusional_growth_mass_change = particulator.attributes[
        "diffusional growth mass change"
    ].get()
    assert abs(diffusional_growth_mass_change) > 0

    # test happens if coagulation added -< check if throws assertion
    #
