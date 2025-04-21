"""test isotope ventilation ratio"""

import pytest
from PySDM import Formulae
from PySDM.physics import isotope_ventilation_ratio, constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestIsotopeVentilationRatio:  # pylint: disable=too-few-public-methods
    @staticmethod
    def test_dimensionality():
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(isotope_ventilation_ratio="Brutsaert1982")
            si = constants_defaults.si
            sut = formulae.isotope_ventilation_ratio.isotope_ventilation_ratio

            # Act
            re = sut(
                ventilation_coefficient=1 * si.dimensionless,
                diffusivity_ratio=1 * si.dimensionless,
            )

            # Assert
            assert re.check(si.dimensionless)
