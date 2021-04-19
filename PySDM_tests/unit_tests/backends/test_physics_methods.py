import PySDM.physics.formulae
from PySDM_tests.backends_fixture import backend
import numpy as np


class TestPhysicsMethods:
    @staticmethod
    def test_temperature_pressure_RH(backend):
        # Arrange
        sut = backend.temperature_pressure_RH
        rhod = backend.Storage.from_ndarray(np.asarray((1,1.1)))
        thd = backend.Storage.from_ndarray(np.asarray((300,301)))
        qv = backend.Storage.from_ndarray(np.asarray((.01,.02)))

        T = backend.Storage.from_ndarray(np.zeros_like(qv))
        p = backend.Storage.from_ndarray(np.zeros_like(qv))
        RH = backend.Storage.from_ndarray(np.zeros_like(qv))

        # Act
        sut(rhod, thd, qv, T, p, RH)

        # Assert
        assert np.amin(RH[:]) != 0 and np.amax(RH[:]) != 0
        assert np.amin(p[:]) != 0 and np.amax(p[:]) != 0
        assert np.amin(T[:]) != 0 and np.amax(T[:]) != 0