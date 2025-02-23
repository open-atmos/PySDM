# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae
from PySDM.physics import si


class TestPhysicsMethods:
    @staticmethod
    def test_temperature_pressure_rh(backend_instance):
        # Arrange
        backend = backend_instance
        sut = backend.temperature_pressure_rh
        rhod = backend.Storage.from_ndarray(np.asarray((1, 1.1)))
        thd = backend.Storage.from_ndarray(np.asarray((300.0, 301)))
        water_vapour_mixing_ratio = backend.Storage.from_ndarray(
            np.asarray((0.01, 0.02))
        )

        T = backend.Storage.from_ndarray(np.zeros_like(water_vapour_mixing_ratio))
        p = backend.Storage.from_ndarray(np.zeros_like(water_vapour_mixing_ratio))
        RH = backend.Storage.from_ndarray(np.zeros_like(water_vapour_mixing_ratio))

        # Act
        sut(
            rhod=rhod,
            thd=thd,
            water_vapour_mixing_ratio=water_vapour_mixing_ratio,
            T=T,
            p=p,
            RH=RH,
        )

        # Assert
        assert 282 * si.K < T.amin() < 283 * si.K
        assert 810 * si.hPa < p.amin() < 830 * si.hPa
        assert 1.12 < RH.amin() < 1.13

    @staticmethod
    @pytest.mark.parametrize("variant", ("LiquidSpheres", "MixedPhaseSpheres"))
    def test_mass_to_volume(backend_class, variant):
        # Arrange
        formulae = Formulae(particle_shape_and_density=variant)
        backend = backend_class(formulae, double_precision=True)
        sut = backend.volume_of_water_mass
        mass = np.asarray([1.0, -1.0])
        mass_in = backend.Storage.from_ndarray(mass)
        volume_out = backend.Storage.from_ndarray(np.zeros_like(mass_in))

        # Act
        sut(volume=volume_out, mass=mass_in)

        # Assert
        assert (mass_in.to_ndarray() == mass).all()
        if variant == "LiquidSpheres":
            assert (
                volume_out.to_ndarray()
                == mass_in.to_ndarray() / formulae.constants.rho_w
            ).all()
        elif variant == "MixedPhaseSpheres":
            assert (
                volume_out.to_ndarray()
                == np.where(
                    mass < 0,
                    mass / formulae.constants.rho_i,
                    mass / formulae.constants.rho_w,
                )
            ).all()
        else:
            raise NotImplementedError()

    @staticmethod
    @pytest.mark.parametrize("variant", ("LiquidSpheres", "MixedPhaseSpheres"))
    def test_volume_to_mass(backend_class, variant):
        # Arrange
        formulae = Formulae(particle_shape_and_density=variant)
        backend = backend_class(formulae, double_precision=True)
        sut = backend.mass_of_water_volume
        volume = np.asarray([1.0, -1.0])
        volume_in = backend.Storage.from_ndarray(volume)
        mass_out = backend.Storage.from_ndarray(np.zeros_like(volume_in))

        # Act
        sut(volume=volume_in, mass=mass_out)

        # Assert
        assert (volume_in.to_ndarray() == volume).all()
        if variant == "LiquidSpheres":
            assert (
                mass_out.to_ndarray()
                == volume_in.to_ndarray() * formulae.constants.rho_w
            ).all()
        elif variant == "MixedPhaseSpheres":
            assert (
                mass_out.to_ndarray()
                == np.where(
                    volume < 0,
                    volume * formulae.constants.rho_i,
                    volume * formulae.constants.rho_w,
                )
            ).all()
        else:
            raise NotImplementedError()
