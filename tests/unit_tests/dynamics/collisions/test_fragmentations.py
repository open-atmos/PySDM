# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.dynamics.collisions.breakup_fragmentations import (
    SLAMS,
    AlwaysN,
    ConstantSize,
    ExponFrag,
    Feingold1988Frag,
    Gaussian,
    Straub2010Nf,
)
from PySDM.environments import Box
from PySDM.physics import si

ARBITRARY_VALUE_BETWEEN_0_AND_1 = 0.5


def dummy_u01(builder, size):
    return builder.particulator.PairwiseStorage.from_ndarray(
        np.full(size, ARBITRARY_VALUE_BETWEEN_0_AND_1)
    )


class TestFragmentations:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize(
        "fragmentation_fn",
        (
            AlwaysN(n=2),
            ExponFrag(scale=1e6 * si.um**3),
            Feingold1988Frag(scale=1e6 * si.um**3),
            Gaussian(mu=2e6 * si.um**3, sigma=1e6 * si.um**3),
            SLAMS(),
            Straub2010Nf(),
        ),
    )
    def test_fragmentation_fn_call(fragmentation_fn, backend_class):
        # arrange
        volume = np.asarray([440.0 * si.um**3, 6660.0 * si.um**3])
        fragments = np.asarray([-1.0])
        builder = Builder(
            volume.size,
            backend_class(
                Formulae(fragmentation_function=fragmentation_fn.__class__.__name__)
            ),
        )
        sut = fragmentation_fn
        sut.vmin = 1 * si.um**3
        sut.register(builder)
        builder.set_environment(Box(dv=None, dt=None))
        _ = builder.build(attributes={"volume": volume, "n": np.ones_like(volume)})

        _PairwiseStorage = builder.particulator.PairwiseStorage
        _Indicator = builder.particulator.PairIndicator
        nf = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        frag_size = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = builder.particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )
        u01 = dummy_u01(builder, fragments.size)

        # act
        sut(nf, frag_size, u01, is_first_in_pair)

        # Assert
        np.testing.assert_array_less([0.99], nf.to_ndarray())
        np.testing.assert_array_less([0.0], frag_size.to_ndarray())

    @staticmethod
    @pytest.mark.parametrize(
        "fragmentation_fn",
        [
            ExponFrag(scale=1 * si.um**3, vmin=6660.0 * si.um**3),
            Feingold1988Frag(scale=1 * si.um**3, vmin=6660.0 * si.um**3),
            Gaussian(mu=2 * si.um**3, sigma=1 * si.um**3, vmin=6660.0 * si.um**3),
            SLAMS(vmin=6660.0 * si.um**3),
            Straub2010Nf(vmin=6660.0 * si.um**3),
            pytest.param(AlwaysN(n=10), marks=pytest.mark.xfail(strict=True)),
        ],
    )
    def test_fragmentation_limiters_vmin(fragmentation_fn, backend_class):
        # arrange
        volume = np.asarray([440.0 * si.um**3, 6660.0 * si.um**3])
        fragments = np.asarray([-1.0])
        builder = Builder(
            volume.size,
            backend_class(
                Formulae(fragmentation_function=fragmentation_fn.__class__.__name__),
                double_precision=True,
            ),
        )
        sut = fragmentation_fn
        sut.register(builder)
        builder.set_environment(Box(dv=None, dt=None))
        _ = builder.build(attributes={"volume": volume, "n": np.ones_like(volume)})

        _PairwiseStorage = builder.particulator.PairwiseStorage
        _Indicator = builder.particulator.PairIndicator
        nf = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        frag_size = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = builder.particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )
        u01 = dummy_u01(builder, fragments.size)

        # act
        sut(nf, frag_size, u01, is_first_in_pair)

        # Assert
        np.testing.assert_array_equal([1.0], nf.to_ndarray())
        np.testing.assert_array_equal(
            [(6660.0 + 440.0) * si.um**3], frag_size.to_ndarray()
        )

    @staticmethod
    @pytest.mark.parametrize(
        "fragmentation_fn",
        [
            ExponFrag(scale=1.0 * si.cm**3),
            Feingold1988Frag(scale=1.0 * si.cm**3),
            Gaussian(mu=1.0 * si.cm**3, sigma=1e6 * si.um**3),
            SLAMS(),
            Straub2010Nf(),
            pytest.param(AlwaysN(n=0.01), marks=pytest.mark.xfail(strict=True)),
        ],
    )
    def test_fragmentation_limiters_vmax(fragmentation_fn, backend_class):
        # arrange
        volume = np.asarray([440.0 * si.um**3, 6660.0 * si.um**3])
        fragments = np.asarray([-1.0])
        builder = Builder(
            volume.size,
            backend_class(
                Formulae(fragmentation_function=fragmentation_fn.__class__.__name__)
            ),
        )
        sut = fragmentation_fn
        sut.vmin = 1 * si.um**3
        sut.register(builder)
        builder.set_environment(Box(dv=None, dt=None))
        _ = builder.build(attributes={"volume": volume, "n": np.ones_like(volume)})

        _PairwiseStorage = builder.particulator.PairwiseStorage
        _Indicator = builder.particulator.PairIndicator
        nf = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        frag_size = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = builder.particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )
        u01 = dummy_u01(builder, fragments.size)

        # act
        sut(nf, frag_size, u01, is_first_in_pair)

        # Assert
        np.testing.assert_array_less([0.999], nf.to_ndarray())
        np.testing.assert_array_less(
            frag_size.to_ndarray(), [(6661.0 + 440.0) * si.um**3]
        )

    @staticmethod
    @pytest.mark.parametrize(
        "fragmentation_fn",
        [
            ExponFrag(scale=1.0 * si.um**3, nfmax=2),
            Feingold1988Frag(scale=1.0 * si.um**3, nfmax=2),
            Gaussian(mu=1.0 * si.um**3, sigma=1e6 * si.um**3, nfmax=2),
            SLAMS(nfmax=2),
            Straub2010Nf(nfmax=2),
            pytest.param(AlwaysN(n=10), marks=pytest.mark.xfail(strict=True)),
        ],
    )
    def test_fragmentation_limiters_nfmax(fragmentation_fn, backend_class):
        # arrange
        volume = np.asarray([440.0 * si.um**3, 6660.0 * si.um**3])
        fragments = np.asarray([-1.0])
        builder = Builder(
            volume.size,
            backend_class(
                Formulae(fragmentation_function=fragmentation_fn.__class__.__name__)
            ),
        )
        sut = fragmentation_fn
        sut.vmin = 1 * si.um**3
        sut.register(builder)
        builder.set_environment(Box(dv=None, dt=None))
        _ = builder.build(attributes={"volume": volume, "n": np.ones_like(volume)})

        _PairwiseStorage = builder.particulator.PairwiseStorage
        _Indicator = builder.particulator.PairIndicator
        nf = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        frag_size = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = builder.particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )
        u01 = dummy_u01(builder, fragments.size)

        # act
        sut(nf, frag_size, u01, is_first_in_pair)

        # Assert
        np.testing.assert_array_less(nf.to_ndarray(), [2.0 + 1e-6])
        np.testing.assert_array_less(
            [((6660.0 + 440.0) / 2 - 1) * si.um**3], frag_size.to_ndarray()
        )

    @staticmethod
    @pytest.mark.parametrize(
        "fragmentation_fn",
        (
            AlwaysN(n=2),
            ExponFrag(scale=1e6 * si.um**3),
            Feingold1988Frag(scale=1e6 * si.um**3),
            Gaussian(mu=2e6 * si.um**3, sigma=1e6 * si.um**3),
            SLAMS(),
            Straub2010Nf(),
        ),
    )
    def test_fragmentation_fn_distribution(
        fragmentation_fn, plot=False
    ):  # pylint: disable=too-many-locals, unnecessary-lambda-assignment
        # arrange

        drop_size_L_diam = 0.4 * si.cm
        drop_size_S_diam = 0.2 * si.cm

        get_volume_from_diam = lambda d: (4 / 3) * np.pi * (d / 2) ** 3

        n = 100
        res = np.empty((n, 2), dtype=np.double)

        backend = CPU(
            Formulae(fragmentation_function=fragmentation_fn.__class__.__name__)
        )
        volume = np.asarray(
            [
                get_volume_from_diam(drop_size_S_diam),
                get_volume_from_diam(drop_size_L_diam),
            ]
        )
        fragments = np.asarray([-1.0])
        builder = Builder(volume.size, backend)
        sut = fragmentation_fn
        sut.vmin = 1 * si.um**3
        sut.register(builder)
        builder.set_environment(Box(dv=None, dt=None))
        _ = builder.build(attributes={"volume": volume, "n": np.ones_like(volume)})

        rns = np.linspace(1e-6, 1 - 1e-6, n)
        for i, rn in enumerate(rns):
            print("i", i)
            start = timer()

            _PairwiseStorage = builder.particulator.PairwiseStorage
            _Indicator = builder.particulator.PairIndicator
            nf = _PairwiseStorage.from_ndarray(
                np.zeros_like(fragments, dtype=np.double)
            )
            frag_size = _PairwiseStorage.from_ndarray(
                np.zeros_like(fragments, dtype=np.double)
            )
            is_first_in_pair = _Indicator(length=volume.size)
            is_first_in_pair.indicator = builder.particulator.Storage.from_ndarray(
                np.asarray([True, False])
            )
            u01 = _PairwiseStorage.from_ndarray(
                np.asarray([rn])
            )  # (np.random.rand(*fragments.shape))
            print("u01", u01.data)

            end = timer()
            print("elapsed time setup", end - start)

            start = timer()

            # act
            sut(nf, frag_size, u01, is_first_in_pair)

            end = timer()
            print("elapsed time sut", end - start)
            print(nf.data)
            print(frag_size.data)
            res[i][0] = nf[0]
            res[i][1] = frag_size[0]

            # Assert
            np.testing.assert_array_less([0.99], nf.to_ndarray())
            np.testing.assert_array_less([0.0], frag_size.to_ndarray())

        res = np.asarray(sorted(res, key=lambda x: x[1], reverse=True))
        print(res[:, 0])
        unique_nfs, nfs_counts = np.unique(res[:, 0], return_counts=True)
        unique_frag_size, frag_sizes_counts = np.unique(res[:, 1], return_counts=True)
        print("nfs", unique_nfs, nfs_counts)
        print("frag_sizes", unique_frag_size, frag_sizes_counts)

        plt.hist(res[:, 0])
        if plot:
            plt.show()

    @staticmethod
    @pytest.mark.parametrize(
        "fragmentation_fn, volume, expected_nf",
        (
            (
                ConstantSize(c=4 * si.um**3),
                np.asarray([400.0 * si.um**3, 600.0 * si.um**3]),
                250,
            ),
            (AlwaysN(n=250), np.asarray([400.0 * si.um**3, 600.0 * si.um**3]), 250),
        ),
    )
    def test_fragmentation_nf_and_frag_size_equals(  # TODO #987
        fragmentation_fn,
        volume,
        expected_nf,
        backend_class=CPU,
    ):
        # arrange
        expected_frag_size = np.sum(volume) / expected_nf

        fragments = np.asarray([-1.0])
        builder = Builder(
            volume.size,
            backend_class(
                Formulae(fragmentation_function=fragmentation_fn.__class__.__name__)
            ),
        )
        sut = fragmentation_fn
        sut.vmin = 1 * si.um**3
        sut.register(builder)
        builder.set_environment(Box(dv=None, dt=None))
        _ = builder.build(attributes={"volume": volume, "n": np.ones_like(volume)})

        _PairwiseStorage = builder.particulator.PairwiseStorage
        _Indicator = builder.particulator.PairIndicator
        nf = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        frag_size = _PairwiseStorage.from_ndarray(np.zeros_like(fragments))
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = builder.particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )
        u01 = _PairwiseStorage.from_ndarray(np.ones_like(fragments))

        # act
        sut(nf, frag_size, u01, is_first_in_pair)

        # Assert
        np.testing.assert_array_equal(nf.to_ndarray(), expected_nf)
        np.testing.assert_array_equal([expected_frag_size], frag_size.to_ndarray())
