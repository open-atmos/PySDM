""" tests for air dynamic viscosity formulae """

import pytest
import numpy as np
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.formulae import _choices, Formulae
from PySDM.physics import air_dynamic_viscosity, constants_defaults, si


class TestAirDynamicViscosity:
    @staticmethod
    @pytest.mark.parametrize("variant", _choices(air_dynamic_viscosity))
    def test_dimensionality(variant):
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(air_dynamic_viscosity=variant)
            cdsi = constants_defaults.si
            sut = formulae.air_dynamic_viscosity.eta_air

            # Act
            re = sut(temperature=300 * cdsi.K)

            # Assert
            assert re.check("[pressure]*[time]")

    @staticmethod
    @pytest.mark.parametrize(
        "T, eta",
        (
            (127.232 * si.K, 0.000010242 * si.Pa * si.s),
            (287.946 * si.K, 0.000017691 * si.Pa * si.s),
            (455.357 * si.K, 0.000024581 * si.Pa * si.s),
            (703.125 * si.K, 0.000033520 * si.Pa * si.s),
            (1011.16 * si.K, 0.000043203 * si.Pa * si.s),
            (1332.59 * si.K, 0.000052328 * si.Pa * si.s),
            (1660.71 * si.K, 0.000059963 * si.Pa * si.s),
            (1962.05 * si.K, 0.000067598 * si.Pa * si.s),
            (2270.09 * si.K, 0.000075047 * si.Pa * si.s),
            (2537.95 * si.K, 0.000082495 * si.Pa * si.s),
            (2812.50 * si.K, 0.000090503 * si.Pa * si.s),
        ),
    )
    def test_vs_digitized_data_from_Zografos_et_al_1987_figure_A5(T, eta):
        formulae = Formulae(air_dynamic_viscosity="ZografosEtAl1987")
        eta_test = formulae.air_dynamic_viscosity.eta_air(T)
        np.testing.assert_approx_equal(eta_test, eta, significant=2.4)
