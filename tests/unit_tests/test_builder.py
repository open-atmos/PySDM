# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import Condensation, Displacement, RelaxedVelocity
from PySDM.environments import Box


class TestBuilder:
    @staticmethod
    def test_build_minimal():
        # arrange
        env = Box(dt=np.nan, dv=np.nan)
        builder = Builder(backend=CPU(), n_sd=1, environment=env)

        # act
        particulator = builder.build(
            products=(),
            attributes={k: np.asarray([0]) for k in ("multiplicity", "volume")},
        )

        # assert
        _ = particulator.attributes

    @staticmethod
    def test_request_attribute():
        # arrange
        env = Box(dt=-1, dv=np.nan)
        builder = Builder(backend=CPU(), n_sd=1, environment=env)
        builder.add_dynamic(Condensation())

        # act
        builder.request_attribute("critical supersaturation")

        # assert
        particulator = builder.build(
            products=(),
            attributes={
                k: np.asarray([1])
                for k in (
                    "multiplicity",
                    "volume",
                    "dry volume",
                    "kappa times dry volume",
                )
            },
        )
        env["T"] = np.nan
        _ = particulator.attributes["critical supersaturation"].to_ndarray()

    @staticmethod
    @pytest.mark.parametrize(
        "dynamics",
        (
            (RelaxedVelocity(), Displacement()),
            (Displacement(), RelaxedVelocity()),
        ),
    )
    def test_order_of_dynamic_registration_does_not_matter_for_attribute_mappings(
        dynamics,
    ):
        # arrange
        builder = Builder(backend=CPU(), n_sd=1, environment=Box(dt=-1, dv=np.nan))
        for dynamic in dynamics:
            builder.add_dynamic(dynamic)

        _ = builder.build(
            products=(),
            attributes={
                k: np.asarray([0])
                for k in ("multiplicity", "volume", "relative fall momentum")
            },
        )

        # act
        assert (
            builder.get_attribute(
                attribute_name="relative fall velocity"
            ).__class__.__name__
            == "RelativeFallVelocity"
        )

        # assert
