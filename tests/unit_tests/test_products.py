# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect
from typing import Tuple
from collections import namedtuple
import sys
import numpy as np
import pytest
from PySDM import products
from PySDM.backends import CPU
from PySDM.products.impl.product import Product
from PySDM.products.impl.rate_product import RateProduct
from PySDM.products import (AqueousMassSpectrum, AqueousMoleFraction, TotalDryMassMixingRatio,
                            ParticleSizeSpectrumPerMass, GaseousMoleFraction,
                            FreezableSpecificConcentration, DynamicWallTime,
                            ParticleSizeSpectrumPerVolume,
                            ParticleVolumeVersusRadiusLogarithmSpectrum,
                            RadiusBinnedNumberAveragedTerminalVelocity,
                            FlowVelocityComponent)

_ARGUMENTS = {
    AqueousMassSpectrum: {'key': 'S_VI', 'dry_radius_bins_edges': (0, np.inf)},
    AqueousMoleFraction: {'key': 'S_VI'},
    TotalDryMassMixingRatio: {'density': 1},
    ParticleSizeSpectrumPerMass: {'radius_bins_edges': (0, np.inf)},
    GaseousMoleFraction: {'key': 'O3'},
    FreezableSpecificConcentration: {'temperature_bins_edges': (0, 300)},
    DynamicWallTime: {'dynamic': 'Condensation'},
    ParticleSizeSpectrumPerVolume: {'radius_bins_edges': (0, np.inf)},
    ParticleVolumeVersusRadiusLogarithmSpectrum: {'radius_bins_edges': (0, np.inf)},
    RadiusBinnedNumberAveragedTerminalVelocity: {'radius_bin_edges': (0, np.inf)},
    FlowVelocityComponent: {'component': 0}
}


@pytest.fixture(params=(
    pytest.param(p[1], id=p[0])
    for p in inspect.getmembers(sys.modules[products.__name__], inspect.isclass)
))
def product(request):
    return request.param


class TestProducts:
    @staticmethod
    # pylint: disable=redefined-outer-name
    def test_instantiate_all(product):
        product(**(_ARGUMENTS[product] if product in _ARGUMENTS else {}))

    @staticmethod
    def test_unit_conversion():
        # arrange
        class SUT(Product):
            def __init__(self, unit='m'):
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
        count = 666
        size = 1

        class SUT(RateProduct):
            def __init__(self, unit='s^-1'):
                super().__init__(unit=unit, name=None, counter='', dynamic=None)

        backend = CPU()
        sut = SUT()
        sut.buffer = np.empty(size)
        sut.particulator = namedtuple('_', ('dt',))(dt=dt)

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
        np.testing.assert_allclose(value1, count / n_steps / dt)
        np.testing.assert_allclose(value2, count / n_steps / dt)


    @staticmethod
    @pytest.mark.parametrize('in_out_pair', (
        ('CPUTime', 'CPU time'),
        ('WallTime', 'wall time')
    ))
    def test_camel_case_to_words(in_out_pair: Tuple[str, str]):
        # arrange
        test_input, expected_output = in_out_pair

        # act
        actual_output = Product._camel_case_to_words(test_input)

        # assert
        assert actual_output == expected_output
