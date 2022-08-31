# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics.collisions.breakup_fragmentations import (
    SLAMS,
    AlwaysN,
    ExponFrag,
    Gaussian,
    Straub2010Nf,
)
from PySDM.dynamics.collisions.breakup_fragmentations.straub2010 import Straub2010Nf
from PySDM.environments import Box
from PySDM.physics import si


class TestFragmentations:
    @staticmethod
    @pytest.mark.parametrize(
        "fragmentation_fn",
        [
            AlwaysN(n=2),
            ExponFrag(scale=1e6 * si.um**3),
            Gaussian(mu=1e6 * si.um**3, sigma=2e6 * si.um**3),
            SLAMS(),
            Straub2010Nf(),
        ],
    )
    def test_fragmentation_fn_call(fragmentation_fn, backend_class=CPU):
        # arrange
        volume = np.asarray([44.0, 666.0])
        fragments = np.asarray([-1.0])
        builder = Builder(volume.size, backend_class())
        sut = fragmentation_fn
        sut.register(builder)
        builder.set_environment(Box(dv=None, dt=None))
        _ = builder.build(attributes={"volume": volume, "n": np.ones_like(volume)})

        _PairwiseStorage = builder.particulator.PairwiseStorage
        _Indicator = builder.particulator.PairIndicator
        nf = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        frag_size = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        is_first_in_pair = _Indicator(length=volume.size)
        u01 = _PairwiseStorage.from_ndarray(np.ones_like(fragments))

        # act
        sut(nf, frag_size, u01, is_first_in_pair)

        print(nf.data, frag_size.data)

        # Assert
        np.testing.assert_array_less([0.99], nf.to_ndarray())
        np.testing.assert_array_less([0.0], frag_size.to_ndarray())
