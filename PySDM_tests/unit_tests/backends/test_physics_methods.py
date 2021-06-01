# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend
from PySDM.physics import si, Formulae
import numpy as np


class TestPhysicsMethods:
    @staticmethod
    def test_temperature_pressure_RH(backend):
        # Arrange
        sut = backend(Formulae()).temperature_pressure_RH
        rhod = backend.Storage.from_ndarray(np.asarray((1, 1.1)))
        thd = backend.Storage.from_ndarray(np.asarray((300., 301)))
        qv = backend.Storage.from_ndarray(np.asarray((.01, .02)))

        T = backend.Storage.from_ndarray(np.zeros_like(qv))
        p = backend.Storage.from_ndarray(np.zeros_like(qv))
        RH = backend.Storage.from_ndarray(np.zeros_like(qv))

        # Act
        sut(rhod, thd, qv, T, p, RH)

        # Assert
        assert 282 * si.K < T.amin() < 283 * si.K
        assert 820 * si.hPa < p.amin() < 830 * si.hPa
        assert 1.10 < RH.amin() < 1.11
