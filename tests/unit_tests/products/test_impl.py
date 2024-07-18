# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect
import sys
from collections import namedtuple

import numpy as np
import pytest

from PySDM import Builder, products
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.products import (
    ActivatedEffectiveRadius,
    ActivatedMeanRadius,
    ActivatedParticleConcentration,
    ActivatedParticleSpecificConcentration,
    AqueousMassSpectrum,
    AqueousMoleFraction,
    AreaStandardDeviation,
    DynamicWallTime,
    FlowVelocityComponent,
    FreezableSpecificConcentration,
    FrozenParticleConcentration,
    FrozenParticleSpecificConcentration,
    GaseousMoleFraction,
    MeanVolumeRadius,
    NumberSizeSpectrum,
    ParcelLiquidWaterPath,
    ParticleSizeSpectrumPerMassOfDryAir,
    ParticleSizeSpectrumPerVolume,
    ParticleVolumeVersusRadiusLogarithmSpectrum,
    RadiusBinnedNumberAveragedTerminalVelocity,
    RadiusStandardDeviation,
    TotalDryMassMixingRatio,
    VolumeStandardDeviation,
)
from PySDM.products.impl.product import Product
from PySDM.products.impl.rate_product import RateProduct

_ARGUMENTS = {
    AqueousMassSpectrum: {"key": "S_VI", "dry_radius_bins_edges": (0, np.inf)},
    AqueousMoleFraction: {"key": "S_VI"},
    TotalDryMassMixingRatio: {"density": 1},
    ParticleSizeSpectrumPerMassOfDryAir: {"radius_bins_edges": (0, np.inf)},
    GaseousMoleFraction: {"key": "O3"},
    FreezableSpecificConcentration: {"temperature_bins_edges": (0, 300)},
    DynamicWallTime: {"dynamic": "Condensation"},
    ParticleSizeSpectrumPerVolume: {"radius_bins_edges": (0, np.inf)},
    ParticleVolumeVersusRadiusLogarithmSpectrum: {"radius_bins_edges": (0, np.inf)},
    RadiusBinnedNumberAveragedTerminalVelocity: {"radius_bin_edges": (0, np.inf)},
    FlowVelocityComponent: {"component": 0},
    FrozenParticleConcentration: {"count_unactivated": True, "count_activated": True},
    FrozenParticleSpecificConcentration: {
        "count_unactivated": True,
        "count_activated": True,
    },
    NumberSizeSpectrum: {"radius_bins_edges": (0, np.inf)},
    ActivatedMeanRadius: {"count_activated": True, "count_unactivated": False},
    ActivatedParticleConcentration: {
        "count_activated": True,
        "count_unactivated": False,
    },
    ActivatedParticleSpecificConcentration: {
        "count_activated": True,
        "count_unactivated": False,
    },
    MeanVolumeRadius: {"count_activated": True, "count_unactivated": False},
    RadiusStandardDeviation: {"count_activated": True, "count_unactivated": False},
    AreaStandardDeviation: {"count_activated": True, "count_unactivated": False},
    VolumeStandardDeviation: {"count_activated": True, "count_unactivated": False},
    ActivatedEffectiveRadius: {"count_activated": True, "count_unactivated": False},
    ParcelLiquidWaterPath: {"count_activated": True, "count_unactivated": False},
}


@pytest.fixture(
    params=(
        pytest.param(p[1], id=p[0])
        for p in inspect.getmembers(sys.modules[products.__name__], inspect.isclass)
    ),
    name="product",
)
def product_fixture(request):
    return request.param


class TestProducts:
    @staticmethod
    def test_instantiate_all(product):
        product(**(_ARGUMENTS[product] if product in _ARGUMENTS else {}))

    @staticmethod
    def test_unit_conversion():
        # arrange
        class SUT(Product):
            def __init__(self, unit="m"):
                super().__init__(unit=unit)

            def _impl(self, **kwargs):
                return 1

        sut = SUT(unit="mm")

        # act
        value = sut.get()

        # assert
        assert value == 1e3

    @staticmethod
    def test_rate_product():
        # arrange
        n_steps = 10
        dt = 44
        dv = 123
        rhod = 1.11
        count = 666
        size = 1

        class SUT(RateProduct):
            def __init__(self, unit="s^-1"):
                super().__init__(unit=unit, name=None, counter="", dynamic=None)

        backend = CPU()
        sut = SUT()
        sut.buffer = np.empty(size)
        sut.particulator = namedtuple("_", ("dt", "mesh", "environment"))(
            dt=dt,
            mesh=namedtuple("Mesh", ("dv",))(dv=dv),
            environment={"rhod": backend.Storage.from_ndarray(np.asarray([rhod]))},
        )

        def set_and_notify():
            sut.counter = backend.Storage.from_ndarray(np.full(size, count))
            for _ in range(n_steps):
                sut.notify()

        # act
        set_and_notify()
        value1 = sut.get()
        set_and_notify()
        value2 = sut.get()

        # assert
        np.testing.assert_allclose(value1, count / n_steps / dt / dv / rhod)
        np.testing.assert_allclose(value2, count / n_steps / dt / dv / rhod)

    @staticmethod
    def test_register_can_be_called_twice_on_r_eff():
        # arrange
        sut = products.EffectiveRadius()
        env = Box(dt=0, dv=0)
        builder = Builder(backend=CPU(), n_sd=0, environment=env)
        sut.register(builder)

        # act
        sut.register(builder)
