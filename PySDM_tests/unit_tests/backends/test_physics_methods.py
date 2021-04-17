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
        assert (RH.data[:] != 0).all()
        assert (p.data[:] != 0).all()
        assert (T.data[:] != 0).all()