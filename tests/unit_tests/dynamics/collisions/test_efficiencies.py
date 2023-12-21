# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import matplotlib.pyplot as plt
import numpy as np
import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.coalescence_efficiencies import (
    Berry1967,
    ConstEc,
    LowList1982Ec,
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
            LowList1982Ec(),
            ConstEb(Eb=0.3),
        ],
    )
    def test_efficiency_fn_call(efficiency, backend_class=CPU):
        # arrange
        volume = np.asarray([440.0 * si.um**3, 6660.0 * si.um**3])
        env = Box(dv=None, dt=None)
        builder = Builder(volume.size, backend_class(), environment=env)
        sut = efficiency
        sut.register(builder)
        _ = builder.build(
            attributes={"volume": volume, "multiplicity": np.ones_like(volume)}
        )

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
        np.testing.assert_array_less(eff.to_ndarray(), [1.0 + 1e-6])

    @staticmethod
    @pytest.mark.parametrize(
        "efficiency",
        [
            Straub2010Ec(),
        ],
    )
    def test_efficiency_dist(efficiency, backend_class=CPU, plot=False):
        # pylint: disable=too-many-locals, unnecessary-lambda-assignment
        # arrange
        n_per = 20

        drop_size_L_diam = np.linspace(0.01, 0.5, n_per) * si.cm
        drop_size_S_diam = np.linspace(0.01, 0.2, n_per) * si.cm

        get_volume_from_diam = lambda d: (4 / 3) * np.pi * (d / 2) ** 3

        res = np.ones((n_per, n_per), dtype=np.double) * -1.0

        for i in range(n_per):
            for j in range(n_per):
                dl = drop_size_L_diam[i]
                ds = drop_size_S_diam[j]
                if dl >= ds:
                    volume = np.asarray(
                        [
                            get_volume_from_diam(ds),
                            get_volume_from_diam(dl),
                        ]
                    )
                    env = Box(dv=None, dt=None)
                    builder = Builder(volume.size, backend_class(), environment=env)
                    sut = efficiency
                    sut.register(builder)
                    _ = builder.build(
                        attributes={
                            "volume": volume,
                            "multiplicity": np.ones_like(volume),
                        }
                    )

                    _PairwiseStorage = builder.particulator.PairwiseStorage
                    _Indicator = builder.particulator.PairIndicator
                    eff = _PairwiseStorage.from_ndarray(np.asarray([-1.0]))
                    is_first_in_pair = _Indicator(length=volume.size)
                    is_first_in_pair.indicator = (
                        builder.particulator.Storage.from_ndarray(
                            np.asarray([True, False])
                        )
                    )

                    # act
                    sut(eff, is_first_in_pair)
                    res[i, j] = eff.data

                    # Assert
                    np.testing.assert_array_less([0.0 - 1e-6], eff.to_ndarray())
                    np.testing.assert_array_less(eff.to_ndarray(), [1.0 + 1e-6])

        (dl, ds) = np.meshgrid(drop_size_L_diam, drop_size_S_diam)
        levels = np.linspace(0.0, 1.0, 11)
        cbar = plt.contourf(dl, ds, res.T, levels=levels, cmap="jet")
        plt.colorbar(cbar)

        if plot:
            plt.show()
