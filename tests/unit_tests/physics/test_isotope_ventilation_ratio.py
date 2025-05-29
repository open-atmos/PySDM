"""test isotope ventilation ratio"""

import numpy as np

from PySDM import Formulae
from PySDM.physics import constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestIsotopeVentilationRatio:  # pylint: disable=too-few-public-methods
    @staticmethod
    def test_dimensionality():
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(isotope_ventilation_ratio="Brutsaert1982")
            si = constants_defaults.si
            sut = formulae.isotope_ventilation_ratio.ratio_heavy_to_light

            # Act
            re = sut(
                ventilation_coefficient=1 * si.dimensionless,
                diffusivity_ratio_heavy_to_light=1 * si.dimensionless,
            )

            # Assert
            assert re.check(si.dimensionless)

    @staticmethod
    def test_neglect_returns_one_ignoring_argument():
        # arrange
        formulae = Formulae(isotope_ventilation_ratio="Neglect")
        sut = formulae.isotope_ventilation_ratio.ratio_heavy_to_light

        # act
        value = sut(np.nan, np.nan)

        # assert
        assert value == 1
