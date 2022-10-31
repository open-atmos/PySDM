# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.coalescence_efficiencies import (
    Berry1967,
    ConstEc,
    SpecifiedEff,
    Straub2010Ec,
)
from PySDM.environments import Box
from PySDM.physics import si


class TestEfficiencies:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize(
        "efficiency",
        [
            Berry1967(),
            ConstEc(Ec=0.5),
            SpecifiedEff(A=0.8, B=0.6),
            Straub2010Ec(),
            ConstEb(Eb=0.3),
        ],
    )
    def test_efficiency_fn_call(efficiency, backend_class=CPU):
        # arrange
        volume = np.asarray([440.0 * si.um**3, 6660.0 * si.um**3])
        builder = Builder(volume.size, backend_class())
        sut = efficiency
        sut.register(builder)
        builder.set_environment(Box(dv=None, dt=None))
        _ = builder.build(attributes={"volume": volume, "n": np.ones_like(volume)})

        _PairwiseStorage = builder.particulator.PairwiseStorage
        _Indicator = builder.particulator.PairIndicator
        eff = _PairwiseStorage.from_ndarray(np.asarray([-1.0]))
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = builder.particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )

        # act
        sut(eff, is_first_in_pair)

        # Assert
        np.testing.assert_array_less([0.0 - 1e-6], eff.to_ndarray())
