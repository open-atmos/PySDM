""" tests of the product-class-generating `make_arbitrary_moment_product` function """

import pytest
import numpy as np

from PySDM.products.size_spectral.arbitrary_moment import make_arbitrary_moment_product
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.physics import si


class TestArbitraryMoment:
    """groups tests to make [de]selecting easier"""

    @staticmethod
    @pytest.mark.parametrize(
        "kwargs, expected_unit",
        (
            (
                {
                    "rank": 0,
                    "attr": "radius",
                    "attr_unit": "m",
                    "skip_division_by_m0": True,
                    "skip_division_by_dv": True,
                },
                "1 dimensionless",
            ),
            (
                {
                    "rank": 1,
                    "attr": "radius",
                    "attr_unit": "m",
                    "skip_division_by_m0": False,
                    "skip_division_by_dv": True,
                },
                "1 meter",
            ),
            (
                {
                    "rank": 6,
                    "attr": "radius",
                    "attr_unit": "m",
                    "skip_division_by_m0": True,
                    "skip_division_by_dv": True,
                },
                "1 meter ** 6",
            ),
            (
                {
                    "rank": 1,
                    "attr": "water mass",
                    "attr_unit": "kg",
                    "skip_division_by_m0": False,
                    "skip_division_by_dv": True,
                },
                "1 kilogram",
            ),
            (
                {
                    "rank": 1,
                    "attr": "water mass",
                    "attr_unit": "kg",
                    "skip_division_by_m0": False,
                    "skip_division_by_dv": True,
                },
                "1 kilogram",
            ),
            (
                {
                    "rank": 1,
                    "attr": "water mass",
                    "attr_unit": "kg",
                    "skip_division_by_m0": False,
                    "skip_division_by_dv": False,
                },
                "1.0 kilogram / meter ** 3",
            ),
        ),
    )
    def test_unit(kwargs, expected_unit):
        """tests if the product unit respects arguments (incl. division by dv)"""
        # arrange
        product_class = make_arbitrary_moment_product(**kwargs)

        # act
        sut = product_class()

        # assert
        assert sut.unit == expected_unit

    @staticmethod
    @pytest.mark.parametrize("skip_division_by_dv", (True, False))
    def test_division_by_dv(skip_division_by_dv):
        """tests, using a single-superdroplet setup, if the volume normalisation logic works"""
        # arrange
        dv = 666 * si.m**3
        product_class = make_arbitrary_moment_product(
            rank=1,
            attr="water mass",
            attr_unit="kg",
            skip_division_by_m0=False,
            skip_division_by_dv=skip_division_by_dv,
        )
        builder = Builder(n_sd=1, backend=CPU(), environment=Box(dv=dv, dt=np.nan))
        particulator = builder.build(
            attributes={
                k: np.ones(builder.particulator.n_sd)
                for k in ("multiplicity", "water mass")
            },
            products=(product_class(name="sut"),),
        )

        # act
        values = particulator.products["sut"].get()

        # assert
        assert values == 1 / (1 if skip_division_by_dv else dv)
