"""
tests for ventliation coefficient implmentation
"""

from matplotlib import pyplot
import numpy as np
import pytest
from PySDM import Formulae
from PySDM.formulae import _choices
from PySDM.physics import ventilation, constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestVentilationCoefficient:
    @staticmethod
    @pytest.mark.parametrize("variant", _choices(ventilation))
    def test_dimensionality(variant):
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(ventilation=variant)
            si = constants_defaults.si
            sut = formulae.ventilation.ventilation_coefficient

            # Act
            re = sut(sqrt_re_times_cbrt_sc=1 * si.dimensionless)

            # Assert
            assert re.check("[]")

    @staticmethod
    @pytest.mark.parametrize(
        "X, fV",
        (
            (0.133, 1.020),
            (0.665, 1.020),
            (1.662, 1.250),
            (3.989, 1.974),
            (6.582, 2.797),
            (9.440, 3.686),
            (11.17, 4.212),
            (13.50, 4.936),
            (16.69, 5.956),
            (20.21, 7.075),
            (23.53, 8.128),
            (27.52, 9.346),
            (30.85, 10.40),
            (34.57, 11.58),
            (38.16, 12.70),
            (41.22, 13.66),
            (44.61, 14.74),
            (48.53, 15.93),
        ),
    )
    def test_vs_digitized_data_from_pruppacher_and_rassmussen_1979_figure_1(X, fV):
        formulae = Formulae(ventilation="PruppacherAndRasmussen1979")
        fV_test = formulae.ventilation.ventilation_coefficient(X)
        np.testing.assert_approx_equal(fV_test, fV, significant=1.95)

    @staticmethod
    def test_neglect_returns_unity():
        formulae = Formulae(ventilation="Neglect")
        X = np.linspace(0, 50)
        assert (formulae.ventilation.ventilation_coefficient(X) == 1).all()

    @staticmethod
    def test_all_variants_similar(plot=False):
        # arrange/act
        X = np.linspace(0, 50)
        coeffs = {
            paper: Formulae(ventilation=paper).ventilation.ventilation_coefficient(X)
            for paper in _choices(ventilation)
            if paper != "Neglect"
        }

        # plot
        for paper, coeff in coeffs.items():
            pyplot.plot(X, coeff, label=paper)
        pyplot.ylabel("Ventilation Coefficient")
        pyplot.xlabel("$Sc^{1/3} Re^{1/2}$")
        pyplot.grid()
        pyplot.legend()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        for paper, coeff in coeffs.items():
            reference = "Froessling1938"
            if paper == reference:
                continue
            assert (abs(1 - coeff / coeffs[reference]) < 0.15).all()

    @staticmethod
    def test_pruppacher_and_rasmussen_1979_finite_value_at_threshold():
        # arrange
        formulae = Formulae(ventilation="PruppacherAndRasmussen1979")
        sut = formulae.ventilation.ventilation_coefficient

        # act
        f = sut(
            sqrt_re_times_cbrt_sc=constants_defaults.PRUPPACHER_RASMUSSEN_1979_XTHRES
        )

        # assert
        assert np.isfinite(f)
        assert 1.2 < f < 1.22
