# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.attributes.physics import Multiplicities
from PySDM.environments import Box

from ...backends_fixture import backend_class

assert hasattr(backend_class, "_pytestfixturefunction")


class TestMultiplicities:
    @staticmethod
    def test_max_multiplicity_value():
        actual_max_multiplicity = Multiplicities.MAX_VALUE
        expected_max_multiplicity = np.iinfo(
            np.int64
        ).max  # TODO #324: switch to uint64?

        assert actual_max_multiplicity == expected_max_multiplicity

    # pylint: disable=redefined-outer-name
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
    def test_max_multiplicity_assignable(backend_class, value):
        # arrange
        n_sd = 1
        builder = Builder(n_sd=n_sd, backend=backend_class())
        builder.set_environment(Box(dt=np.nan, dv=np.nan))

        # act
        particulator = builder.build(
            attributes={
                "n": np.full((n_sd,), value),
                "volume": np.full((n_sd,), np.nan),
            }
        )

        # assert
        assert particulator.attributes["n"].data[:] == [value]
