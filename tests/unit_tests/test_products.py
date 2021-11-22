import inspect
from typing import Tuple
import sys
import numpy as np
import pytest
from PySDM import products
from PySDM.products.impl.product import Product
from PySDM.products import (AqueousMassSpectrum, AqueousMoleFraction, TotalDryMassMixingRatio,
                            ParticleSizeSpectrumPerMass, GaseousMoleFraction,
                            FreezableSpecificConcentration, DynamicWallTime,
                            ParticleSizeSpectrumPerVolume, ParticlesVolumeSpectrum)

_ARGUMENTS = {
    AqueousMassSpectrum: {'key': 'S_VI', 'dry_radius_bins_edges': (0, np.inf)},
    AqueousMoleFraction: {'key': 'S_VI'},
    TotalDryMassMixingRatio: {'density': 1},
    ParticleSizeSpectrumPerMass: {'radius_bins_edges': (0, np.inf)},
    GaseousMoleFraction: {'key': 'O3'},
    FreezableSpecificConcentration: {'temperature_bins_edges': (0, 300)},
    DynamicWallTime: {'dynamic': 'Condensation'},
    ParticleSizeSpectrumPerVolume: {'radius_bins_edges': (0, np.inf)},
    ParticlesVolumeSpectrum: {'radius_bins_edges': (0, np.inf)}
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
