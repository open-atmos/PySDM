# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import Condensation, Displacement
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
    def test_replace_dynamic():
        # arrange
        env = Box(dt=-1, dv=np.nan)
        builder = Builder(backend=CPU(), n_sd=1, environment=env)
        builder.add_dynamic(Displacement(adaptive=False))

        # act
        builder.replace_dynamic(Displacement(adaptive=True))

        # assert
        assert (
            "Condensation"
            not in builder.build(
                products=(),
                attributes={k: np.asarray([0]) for k in ("multiplicity", "volume")},
            ).dynamics
        )
