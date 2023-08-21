# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import numpy as np
import pytest

from PySDM import Builder
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.products import RadiusStandardDeviation

# def area_std(r, n, mask):
#     nt = r.shape[1]
#     n_tot = np.sum(np.where(mask, np.array(n), 0), axis=0)
#     r_act = np.where(mask, np.array(r), np.nan)
#     n_act=np.where(mask, np.array(n), np.nan)
#     r_sq = np.multiply(r_act, r_act)
#     area_std = np.full(nt, np.nan)
#     for i in range(nt):
#         if n_tot[i] > 0:
#             area_std[i] = np.sqrt(
#                 np.cov(r_sq[:, i][~np.isnan(r_sq[:, i])],
#                 fweights=n_act[:, i][~np.isnan(n_act[:, i])]))
#     return area_std


class TestSizeStandardDeviation:
    @staticmethod
    @pytest.mark.parametrize(
        "r,n",
        (
            ([1 * si.um], [1]),
            ([1 * si.um, 10 * si.um], [1e3, 1e7]),
        ),
    )
    def test_radius_standard_deviation(backend_class, r, n):
        # arrange
        name = "stdev"
        builder = Builder(n_sd=len(n), backend=backend_class(double_precision=True))

        volume = builder.formulae.trivia.volume(np.asarray(r))
        dry_volume = 0.01 * volume
        kappa = 1

        builder.set_environment(Box(dt=np.nan, dv=np.nan))
        particulator = builder.build(
            attributes={
                "n": np.asarray(n),
                "volume": volume,
                "dry volume": dry_volume,
                "dry volume organic": np.full_like(r, 0),
                "kappa times dry volume": kappa * dry_volume,
            },
            products=(
                RadiusStandardDeviation(
                    name=name, count_activated=True, count_unactivated=True
                ),
            ),
        )

        particulator.environment["T"] = 300 * si.K

        expected = 0 if len(n) == 1 else np.sqrt(np.cov(r, fweights=n))

        # act
        cell_id = 0
        actual = particulator.products[name].get()[cell_id]

        # assert
        np.testing.assert_approx_equal(actual, expected, significant=3)
