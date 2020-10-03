"""
Created at 2019
"""

from PySDM.physics import constants, formulae
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestFormulae:

    @staticmethod
    def test_pvs():
        with DimensionalAnalysis():
            # Arrange
            si = constants.si
            sut = formulae.pvs
            T = 300 * si.kelvins

            # Act
            pvs = sut(T)

            # Assert
            assert pvs.units == si.hectopascals

    @staticmethod
    def test_r_cr():
        with DimensionalAnalysis():
            # Arrange
            si = constants.si
            sut = formulae.r_cr

            kp = .5
            rd = .1 * si.micrometre
            T = 300 * si.kelvins

            # Act
            r_cr = sut(kp, rd, T)

            # Assert
            assert r_cr.to_base_units().units == si.metres

    @staticmethod
    def test_lv():
        with DimensionalAnalysis():
            # Arrange
            si = constants.si
            T = 300 * si.kelvins

            # Act
            latent_heat = formulae.lv(T)

            # Assert
            assert latent_heat.check('[energy]/[mass]')

