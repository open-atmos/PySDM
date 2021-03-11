from PySDM.physics.constants import ROOM_TEMP
from PySDM.backends.numba.storage import Storage
from PySDM.dynamics.aqueous_chemistry.aqueous_chemistry import equilibrate_pH
from PySDM.physics.constants import convert_to
from PySDM.physics import formulae as phys
from PySDM.physics import si
import numpy as np


class TestAqueousChemistry:
    @staticmethod
    def test_equilibrate_pH_pure_water():
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


    @staticmethod
    def test_equilibrate_pH_non_trivial():
        from chempy import Equilibrium
        from chempy.chemistry import Species
        water_autop = Equilibrium({'H2O'}, {'H+', 'OH-'}, 10 ** -14)  # unit "molar" assumed
        ammonia_prot = Equilibrium({'NH4+'}, {'NH3', 'H+'}, 10 ** -9.24)  # same here

        # nitric_prot = Equilibrium({''})
        # # ("HNO3(aq) = H+ + NO3-", 15.4, 0),
        # "K_HNO3": EqConst(15.4 * M, 0 * dT_u),
        # # ("H2SO3(aq) = H+ + HSO3-", 1.54*10**-2 * KU, 1960),
        # "K_SO2": EqConst(1.3e-2 * M, 1960 * dT_u),
        # # ("NH4+ = NH3(aq) + H+", 10**-9.25 * M, 0),
        # "K_NH3": EqConst(1.7e-5 * M, -450 * dT_u),
        # # ("H2CO3(aq) = H+ + HCO3-", 4.3*10**-7 * KU, -1000),
        # "K_CO2": EqConst(4.3e-7 * M, -1000 * dT_u),
        # # ("HSO3- = H+ + SO3-2", 6.6*10**-8 * KU, 1500),
        # "K_HSO3": EqConst(6.6e-8 * M, 1500 * dT_u),
        # # ("HCO3- = H+ + CO3-2", 4.68*10**-11 * KU, -1760),
        # "K_HCO3": EqConst(4.68e-11 * M, -1760 * dT_u),
        # # ("HSO4- = H+ + SO4-2", 1.2*10**-2 * KU, 2720),
        # "K_HSO4": EqConst(1.2e-2 * M, 2720 * dT_u),

        pass

    @staticmethod
    def test_oxidize():
        from chempy import ReactionSystem  # The rate constants below are arbtrary
        rsys = ReactionSystem.from_string("""2 Fe+2 + H2O2 -> 2 Fe+3 + 2 OH-; 42
            2 Fe+3 + H2O2 -> 2 Fe+2 + O2 + 2 H+; 17
            H+ + OH- -> H2O; 1e10   
            H2O -> H+ + OH-; 1e-4""")  # "[H2O]" = 1.0 (actually 55.4 at RT)

    @staticmethod
    def test_henry():
        from chempy.henry import Henry
        kH_O2 = Henry(1.2e-3, 1800, ref='carpenter_1966')

        from PySDM.dynamics.aqueous_chemistry.support import HENRY_CONST

