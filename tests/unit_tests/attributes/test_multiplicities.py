# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.attributes.physics import Multiplicities
from PySDM.environments import Box


class TestMultiplicities:
    @staticmethod
    def test_max_multiplicity_value():
        actual_max_multiplicity = Multiplicities.MAX_VALUE
        expected_max_multiplicity = np.iinfo(
            np.int64
        ).max  # TODO #324: switch to uint64?

        assert actual_max_multiplicity == expected_max_multiplicity

    @staticmethod
    @pytest.mark.parametrize(
        "value",
        (
            Multiplicities.MAX_VALUE,
            pytest.param(
                Multiplicities.MAX_VALUE + 1, marks=pytest.mark.xfail(strict=True)
            ),
        ),
    )
    def test_max_multiplicity_assignable(backend_instance, value):
        # arrange
        n_sd = 1
        env = Box(dt=np.nan, dv=np.nan)
        builder = Builder(n_sd=n_sd, backend=backend_instance, environment=env)

        # act
        particulator = builder.build(
            attributes={
                "multiplicity": np.full((n_sd,), value),
                "volume": np.full((n_sd,), np.nan),
            }
        )

        # assert
        assert particulator.attributes["multiplicity"].to_ndarray() == [value]
