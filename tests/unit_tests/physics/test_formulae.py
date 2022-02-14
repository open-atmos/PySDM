# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

from PySDM.physics import constants_defaults, saturation_vapour_pressure, latent_heat
from PySDM.formulae import Formulae, _choices
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestFormulae:
    @staticmethod
    @pytest.mark.parametrize('opt', _choices(saturation_vapour_pressure))
    def test_pvs(opt):
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(saturation_vapour_pressure=opt)
            si = constants_defaults.si
            sut = formulae.saturation_vapour_pressure.pvs_Celsius
            T = 300 * si.kelvins

            # Act
            pvs = sut(T)

            # Assert
            assert pvs.units == si.hectopascals

    @staticmethod
    def test_r_cr():
        with DimensionalAnalysis():
            # Arrange
            si = constants_defaults.si
            formulae = Formulae()
            sut = formulae.hygroscopicity.r_cr

            kp = .5
            rd = .1 * si.micrometre
            T = 300 * si.kelvins
            sgm = constants_defaults.sgm_w

            # Act
            r_cr = sut(kp, rd**3, T, sgm)

            # Assert
            assert r_cr.to_base_units().units == si.metres

    @staticmethod
    @pytest.mark.parametrize('opt', _choices(latent_heat))
    def test_lv(opt):
        with DimensionalAnalysis():
            # Arrange
            si = constants_defaults.si
            T = 300 * si.kelvins

            formulae = Formulae(latent_heat=opt)
            sut = formulae.latent_heat.lv

            # Act
            latent_heat = sut(T)

            # Assert
            assert latent_heat.check('[energy]/[mass]')

    @staticmethod
    def test___str__():
        # Arrange
        sut = Formulae()

        # Act
        result = str(sut)

        # Assert
        assert len(result) > 0
