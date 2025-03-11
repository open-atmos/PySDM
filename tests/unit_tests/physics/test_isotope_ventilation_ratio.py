"""test isotope ventilation ratio"""

import pytest
from PySDM import Formulae
from PySDM.formulae import _choices
from PySDM.physics import isotope_ventilation_ratio, constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestIsotopeVentilationRatio:  # pylint disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize("variant", _choices(isotope_ventilation_ratio))
    def test_dimensionality(variant):
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(isotope_ventilation_ratio=variant)
            si = constants_defaults.si
            sut = formulae.isotope_ventilation_ratio.isotope_ventilation_coefficient

            # Act
            re = sut(
                sqrt_re_times_cbrt_sc=1 * si.dimensionless,
                diffusivity_ratio=1 * si.dimensionless,
            )

            # Assert
            assert re.check(si.dimensionless)
