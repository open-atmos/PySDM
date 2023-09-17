# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM.physics import si


class TestPhysicsMethods:  # pylint: disable=too-few-public-methods
    @staticmethod
    def test_temperature_pressure_RH(backend_class):
        # Arrange
        backend = backend_class()
        sut = backend.temperature_pressure_RH
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
        assert 820 * si.hPa < p.amin() < 830 * si.hPa
        assert 1.10 < RH.amin() < 1.11
