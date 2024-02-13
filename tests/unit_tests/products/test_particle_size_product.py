# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.products import (
    ActivatedMeanRadius,
    AreaStandardDeviation,
    MeanVolumeRadius,
    RadiusStandardDeviation,
)

TRIVIA = Formulae().trivia
NAME = "tested product"
KAPPA = 1
CELL_ID = 0


def radius_std(r, n, mask):
    n_tot = np.sum(np.where(mask, n, 0), axis=0)
    r_act = np.where(mask, r, np.nan)
    n_act = np.where(mask, n, np.nan)
    std = (
        np.sqrt(np.cov(r_act[~np.isnan(r_act)], fweights=n_act[~np.isnan(n_act)]))
        if n_tot > 1
        else 0
    )
    return std


def filtered_mean(attribute, multiplicity, mask):
    n_dot_a = np.multiply(np.where(mask, multiplicity, 0), attribute)
    n_tot = np.sum(np.where(mask, multiplicity, 0), axis=0)
    mean = np.sum(n_dot_a, axis=0) / n_tot if n_tot > 0 else 0
    return mean


def r_vol_mean(r, n, mask):
    volume = TRIVIA.volume(radius=r)
    mean_volume = filtered_mean(volume, n, mask)
    return TRIVIA.radius(volume=mean_volume)


def area_std(r, n, mask):
    n_tot = np.sum(np.where(mask, n, 0), axis=0)
    n_act = np.where(mask, n, np.nan)
    r_sq = np.where(mask, np.multiply(r, r), np.nan) * 4 * np.pi
    std = (
        np.sqrt(np.cov(r_sq[~np.isnan(r_sq)], fweights=n_act[~np.isnan(n_act)]))
        if n_tot > 1
        else 0
    )
    return std


@pytest.mark.parametrize(
    "r,n",
    (
        ([1 * si.um], [1]),
        ([1 * si.um, 10 * si.um], [1e3, 1e7]),
        ([0.01 * si.um, 0.1 * si.um], [1e3, 1e7]),
    ),
)
@pytest.mark.parametrize("count_unactivated", (True, False))
@pytest.mark.parametrize("count_activated", (True, False))
@pytest.mark.parametrize(
    "product_class, validation_fun",
    (
        (RadiusStandardDeviation, radius_std),
        (ActivatedMeanRadius, filtered_mean),
        (AreaStandardDeviation, area_std),
        (MeanVolumeRadius, r_vol_mean),
    ),
)
# pylint: disable=too-many-arguments
def test_particle_size_product(
    backend_class,
    r,
    n,
    count_unactivated,
    count_activated,
    product_class,
    validation_fun,
):
    # arrange
    builder = Builder(
        n_sd=len(n),
        backend=backend_class(double_precision=True),
        environment=Box(dt=np.nan, dv=np.nan),
    )
    volume = builder.formulae.trivia.volume(np.asarray(r))
    dry_volume = np.full_like(volume, (0.01 * si.um) ** 3)

    builder.request_attribute("critical volume")
    particulator = builder.build(
        attributes={
            "multiplicity": np.asarray(n),
            "volume": volume,
            "dry volume": dry_volume,
            "kappa times dry volume": KAPPA * dry_volume,
        },
        products=(
            product_class(
                name=NAME,
                count_activated=count_activated,
                count_unactivated=count_unactivated,
            ),
        ),
    )

    particulator.environment["T"] = 300 * si.K

    crit_volume = particulator.attributes["critical volume"].to_ndarray()
    if count_activated and not count_unactivated:
        mask = volume > crit_volume
    elif not count_activated and count_unactivated:
        mask = volume < crit_volume
    elif count_activated and count_unactivated:
        mask = np.full_like(volume, True)
    else:
        mask = np.full_like(volume, False)

    expected = validation_fun(np.asarray(r), np.asarray(n), mask)

    # act
    actual = particulator.products[NAME].get()[CELL_ID]

    # assert
    np.testing.assert_almost_equal(actual, expected, decimal=10)
