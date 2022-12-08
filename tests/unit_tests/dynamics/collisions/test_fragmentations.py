# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.dynamics.collisions.breakup_fragmentations import (
    SLAMS,
    AlwaysN,
    ExponFrag,
    Feingold1988Frag,
    Gaussian,
    Straub2010Nf,
)
from PySDM.environments import Box
from PySDM.physics import si

from ....backends_fixture import backend_class

assert hasattr(backend_class, "_pytestfixturefunction")


class TestFragmentations:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize(
        "fragmentation_fn",
        [
            AlwaysN(n=2),
            ExponFrag(scale=1e6 * si.um**3),
            Feingold1988Frag(scale=1e6 * si.um**3),
            Gaussian(mu=2e6 * si.um**3, sigma=1e6 * si.um**3),
            SLAMS(),
            Straub2010Nf(),
        ],
    )
    def test_fragmentation_fn_call(
        fragmentation_fn, backend_class
    ):  # pylint: disable=redefined-outer-name
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
        u01 = _PairwiseStorage.from_ndarray(np.ones_like(fragments))

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
    def test_fragmentation_limiters_vmin(
        fragmentation_fn, backend_class
    ):  # pylint: disable=redefined-outer-name
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
        u01 = _PairwiseStorage.from_ndarray(np.ones_like(fragments))

        # act
        sut(nf, frag_size, u01, is_first_in_pair)

        # Assert
        np.testing.assert_array_equal([(440.0 + 6660.0) / 6660.0], nf.to_ndarray())
        np.testing.assert_array_equal([6660.0 * si.um**3], frag_size.to_ndarray())

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
    def test_fragmentation_limiters_vmax(
        fragmentation_fn, backend_class
    ):  # pylint: disable=redefined-outer-name
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
        u01 = _PairwiseStorage.from_ndarray(np.ones_like(fragments))

        # act
        sut(nf, frag_size, u01, is_first_in_pair)

        # Assert
        np.testing.assert_array_less([(440.0 + 6660.0) / 6661.0], nf.to_ndarray())
        np.testing.assert_array_less(frag_size.to_ndarray(), [6661.0 * si.um**3])

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
    def test_fragmentation_limiters_nfmax(
        fragmentation_fn, backend_class
    ):  # pylint: disable=redefined-outer-name
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
        u01 = _PairwiseStorage.from_ndarray(np.ones_like(fragments))

        # act
        sut(nf, frag_size, u01, is_first_in_pair)

        # Assert
        np.testing.assert_array_less(nf.to_ndarray(), [2.0 + 1e-6])
        np.testing.assert_array_less(
            [((6660.0 + 440.0) / 2 - 1) * si.um**3], frag_size.to_ndarray()
        )
