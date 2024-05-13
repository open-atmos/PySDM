"""
tests for ventilation coefficient implementation
"""

from matplotlib import pyplot
import numpy as np
import pytest
from PySDM import Formulae
from PySDM.formulae import _choices
from PySDM.physics import ventilation, constants_defaults, in_unit
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

    @staticmethod
    @pytest.mark.parametrize(
        "diameter_cm, first_factor_cm, temperature",
        (
            (0.01, 0.086, constants_defaults.T0),
            (0.05, 1.01, constants_defaults.T0),
            (0.1, 2.90, constants_defaults.T0),
            (0.2, 8.80, constants_defaults.T0),
            (0.3, 20.1, constants_defaults.T0),
            (0.01, 0.082, constants_defaults.T0 + 10),
            (0.05, 0.99, constants_defaults.T0 + 10),
            (0.1, 2.80, constants_defaults.T0 + 10),
            (0.2, 8.50, constants_defaults.T0 + 10),
            (0.3, 19.3, constants_defaults.T0 + 10),
            (0.01, 0.079, constants_defaults.T0 + 20),
            (0.05, 0.97, constants_defaults.T0 + 20),
            (0.1, 2.80, constants_defaults.T0 + 20),
            (0.2, 8.30, constants_defaults.T0 + 20),
            (0.3, 18.5, constants_defaults.T0 + 20),
            (0.01, 0.076, constants_defaults.T0 + 30),
            (0.05, 0.96, constants_defaults.T0 + 30),
            (0.1, 2.70, constants_defaults.T0 + 30),
            (0.2, 8.10, constants_defaults.T0 + 30),
            (0.3, 17.8, constants_defaults.T0 + 30),
            (0.01, 0.073, constants_defaults.T0 + 40),
            (0.05, 0.94, constants_defaults.T0 + 40),
            (0.1, 2.70, constants_defaults.T0 + 40),
            (0.2, 8.00, constants_defaults.T0 + 40),
            (0.3, 17.2, constants_defaults.T0 + 40),
        ),
    )
    def test_table_1_from_kinzer_and_gunn_1951(
        diameter_cm, temperature, first_factor_cm
    ):
        # arrange
        formulae = Formulae(
            ventilation="PruppacherAndRasmussen1979", terminal_velocity="RogersYau"
        )
        si = constants_defaults.si
        const = formulae.constants
        air_density = 1 * si.kg / si.m**3
        radius = diameter_cm * si.cm / 2
        dynamic_viscosity = formulae.air_dynamic_viscosity.eta_air(temperature)
        Sc = formulae.trivia.air_schmidt_number(
            dynamic_viscosity, const.D0, air_density
        )
        Re = formulae.particle_shape_and_density.reynolds_number(
            radius,
            formulae.terminal_velocity.v_term(radius),
            dynamic_viscosity,
            air_density,
        )
        vent_coeff = formulae.ventilation.ventilation_coefficient(
            formulae.trivia.sqrt_re_times_cbrt_sc(Re, Sc)
        )
        np.testing.assert_allclose(
            actual=in_unit(4 * np.pi * radius * vent_coeff, si.cm),
            desired=first_factor_cm,
            rtol=0.31,
        )
