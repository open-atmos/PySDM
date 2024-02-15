""" tests different formulations of effective radius product """

import numpy as np

from PySDM import Builder
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.products import EffectiveRadius, ActivatedEffectiveRadius


def test_effective_radii(backend_class):
    # arrange
    env = Box(dt=np.nan, dv=np.nan)
    wet_radii = np.asarray([0.01 * si.um, 0.05 * si.um, 0.09 * si.um, 1 * si.um])
    dry_radii = np.asarray([0.009 * si.um] * len(wet_radii))
    kappa = 1.666

    builder = Builder(
        n_sd=len(wet_radii),
        backend=backend_class(double_precision=True),
        environment=env,
    )
    dry_volume = builder.formulae.trivia.volume(radius=dry_radii)
    env["T"] = 300 * si.K
    particulator = builder.build(
        attributes={
            "water mass": builder.formulae.particle_shape_and_density.radius_to_mass(
                wet_radii
            ),
            "dry volume": dry_volume,
            "kappa times dry volume": kappa * dry_volume,
            "multiplicity": np.asarray([1] * len(wet_radii)),
        },
        products=(
            ActivatedEffectiveRadius(
                name="a", count_unactivated=False, count_activated=False
            ),
            ActivatedEffectiveRadius(
                name="b", count_unactivated=True, count_activated=False
            ),
            ActivatedEffectiveRadius(
                name="c", count_activated=True, count_unactivated=True
            ),
            EffectiveRadius(name="d", radius_range=(0.5 * si.um, np.inf)),
        ),
    )

    # act
    sut = {k: product.get()[0] for k, product in particulator.products.items()}

    # assert
    assert np.isnan(sut["a"])
    assert min(wet_radii) < sut["b"]
    assert sut["b"] < sut["c"]
    assert sut["c"] < sut["d"]
    assert sut["d"] < max(wet_radii)
