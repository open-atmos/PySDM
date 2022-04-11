# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

from PySDM.formulae import Formulae, _choices
from PySDM.physics import (
    constants_defaults,
    diffusion_kinetics,
    diffusion_thermics,
    latent_heat,
    saturation_vapour_pressure,
)
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestFormulae:
    @staticmethod
    @pytest.mark.parametrize("opt", _choices(saturation_vapour_pressure))
    def test_pvs_liq(opt):
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(saturation_vapour_pressure=opt)
            si = constants_defaults.si
            sut = formulae.saturation_vapour_pressure.pvs_Celsius
            T = 300 * si.kelvins

            # Act
            pvs = sut(T)

            # Assert
            assert pvs.check("[pressure]")

    @staticmethod
    @pytest.mark.parametrize("opt", _choices(saturation_vapour_pressure))
    def test_pvs_ice(opt):
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae(saturation_vapour_pressure=opt)
            si = constants_defaults.si
            sut = formulae.saturation_vapour_pressure.ice_Celsius
            T = 250 * si.kelvins

            # Act
            pvs = sut(T)

            # Assert
            assert pvs.check("[pressure]")

    @staticmethod
    def test_r_cr():
        with DimensionalAnalysis():
            # Arrange
            si = constants_defaults.si
            formulae = Formulae()
            sut = formulae.hygroscopicity.r_cr

            kp = 0.5
            rd = 0.1 * si.micrometre
            T = 300 * si.kelvins
            sgm = constants_defaults.sgm_w

            # Act
            r_cr = sut(kp, rd**3, T, sgm)

            # Assert
            assert r_cr.to_base_units().units == si.metres

    @staticmethod
    @pytest.mark.parametrize("opt", _choices(latent_heat))
    def test_lv(opt):
        with DimensionalAnalysis():
            # Arrange
            si = constants_defaults.si
            T = 300 * si.kelvins

            formulae = Formulae(latent_heat=opt)
            sut = formulae.latent_heat.lv

            # Act
            lv = sut(T)

            # Assert
            assert lv.check("[energy]/[mass]")

    @staticmethod
    @pytest.mark.parametrize("opt", _choices(diffusion_thermics))
    def test_thermal_conductivity_temperature_dependence(opt):
        with DimensionalAnalysis():
            # Arrange
            si = constants_defaults.si
            T = 300 * si.kelvins
            p = 1000 * si.hPa

            formulae = Formulae(diffusion_thermics=opt)
            sut = formulae.diffusion_thermics.K

            # Act
            thermal_conductivity = sut(T, p)

            # Assert
            assert thermal_conductivity.check("[power]/[length]/[temperature]")

    @staticmethod
    @pytest.mark.parametrize("opt", _choices(diffusion_kinetics))
    def test_thermal_conductivity_radius_dependence(opt):
        with DimensionalAnalysis():
            # Arrange
            si = constants_defaults.si
            r = 1 * si.um
            lmbd = 0.1 * si.um

            formulae = Formulae(diffusion_kinetics=opt)
            sut = formulae.diffusion_kinetics.K

            # Act
            thermal_conductivity = sut(constants_defaults.K0, r, lmbd)

            # Assert
            assert thermal_conductivity.check("[power]/[length]/[temperature]")

    @staticmethod
    @pytest.mark.parametrize("opt", _choices(diffusion_thermics))
    def test_vapour_diffusivity_temperature_dependence(opt):
        with DimensionalAnalysis():
            # Arrange
            si = constants_defaults.si
            T = 300 * si.kelvins
            p = 1000 * si.hPa

            formulae = Formulae(diffusion_thermics=opt)
            sut = formulae.diffusion_thermics.D

            # Act
            vpour_diffusivity = sut(T, p)

            # Assert
            assert vpour_diffusivity.check("[area]/[time]")

    @staticmethod
    @pytest.mark.parametrize("opt", _choices(diffusion_kinetics))
    def test_vapour_diffusivity_radius_dependence(opt):
        with DimensionalAnalysis():
            # Arrange
            si = constants_defaults.si
            r = 1 * si.um
            lmbd = 0.1 * si.um

            formulae = Formulae(diffusion_kinetics=opt)
            sut = formulae.diffusion_kinetics.D

            # Act
            vpour_diffusivity = sut(constants_defaults.D0, r, lmbd)

            # Assert
            assert vpour_diffusivity.check("[area]/[time]")

    @staticmethod
    def test___str__():
        # Arrange
        sut = Formulae()

        # Act
        result = str(sut)

        # Assert
        assert len(result) > 0
