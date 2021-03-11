from PySDM.physics.constants import ROOM_TEMP
from PySDM.backends.numba.storage import Storage
from PySDM.dynamics.aqueous_chemistry.aqueous_chemistry import equilibrate_pH
from PySDM.physics.constants import convert_to
from PySDM.physics import formulae as phys
from PySDM.physics import si
import numpy as np


class TestAqueousChemistry():
    @staticmethod
    def test_equilibrate_pH():
        # Arrange
        n_sd = 2
        super_droplet_ids = np.arange(n_sd)
        particles = {
            "conc_N_mIII": Storage.from_ndarray(np.zeros(n_sd)),
            "conc_N_V": Storage.from_ndarray(np.zeros(n_sd)),
            "conc_C_IV": Storage.from_ndarray(np.zeros(n_sd)),
            "conc_S_IV": Storage.from_ndarray(np.zeros(n_sd)),
            "conc_S_VI": Storage.from_ndarray(np.zeros(n_sd)),
            "volume": Storage.from_ndarray(np.full(n_sd, phys.volume(radius=1*si.um))),
            "moles_H": Storage.from_ndarray(np.zeros(n_sd)),
        }
        env_T = ROOM_TEMP

        # Act
        equilibrate_pH(super_droplet_ids, particles, env_T)

        # Assert
        volume = particles["volume"].data
        convert_to(volume, si.litre)
        np.testing.assert_allclose(-np.log10(particles["moles_H"].data / volume), 7)

