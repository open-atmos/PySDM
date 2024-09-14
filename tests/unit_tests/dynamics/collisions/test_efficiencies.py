# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from matplotlib import pyplot
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


class TestEfficiencies:
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
        builder = Builder(
            volume.size, backend_class(), environment=Box(dv=None, dt=None)
        )
        sut = efficiency
        sut.register(builder)
        particulator = builder.build(
            attributes={"volume": volume, "multiplicity": np.ones_like(volume)}
        )

        eff = particulator.PairwiseStorage.from_ndarray(np.asarray([-1.0]))
        is_first_in_pair = particulator.PairIndicator(length=volume.size)
        is_first_in_pair.indicator = particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )

        # act
        sut(eff, is_first_in_pair)
        values = eff.to_ndarray()

        # Assert
        assert np.min(values) >= 0
        assert np.max(values) <= 1

    @staticmethod
    @pytest.mark.parametrize(
        "sut",
        [
            Straub2010Ec(),
        ],
    )
    def test_efficiency_dist(sut, backend_class=CPU, plot=False):
        # arrange
        n_per = 20
        n_sd = 2

        drop_size_L_diam = np.linspace(0.01, 0.5, n_per) * si.cm
        drop_size_S_diam = np.linspace(0.01, 0.2, n_per) * si.cm

        builder = Builder(n_sd, backend_class(), environment=Box(dv=None, dt=None))
        sut.register(builder)
        particulator = builder.build(
            attributes={
                "volume": np.full(shape=n_sd, fill_value=np.nan),
                "multiplicity": np.ones(n_sd),
            }
        )

        eff = particulator.PairwiseStorage.from_ndarray(np.asarray([-1.0]))
        is_first_in_pair = particulator.PairIndicator(length=n_sd)
        is_first_in_pair.indicator = particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )

        radius_to_mass = particulator.formulae.particle_shape_and_density.radius_to_mass

        # act
        res = np.full(shape=(n_per, n_per), fill_value=np.nan)
        for i in range(n_per):
            for j in range(n_per):
                if drop_size_L_diam[i] >= drop_size_S_diam[j]:
                    particulator.attributes["water mass"].data.data[:] = np.asarray(
                        [
                            radius_to_mass(drop_size_S_diam[j] / 2),
                            radius_to_mass(drop_size_L_diam[i] / 2),
                        ]
                    )
                    particulator.attributes.mark_updated("water mass")
                    sut(eff, is_first_in_pair)
                    res[i, j] = eff.data

        # plot
        pyplot.colorbar(
            pyplot.contourf(
                *np.meshgrid(drop_size_L_diam, drop_size_S_diam),
                res.T,
                levels=np.linspace(0.0, 1.0, 11),
                cmap="jet"
            )
        )

        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert np.nanmax(res) <= 1
        assert np.nanmin(res) >= 0
