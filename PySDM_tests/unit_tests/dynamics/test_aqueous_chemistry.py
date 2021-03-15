from PySDM.physics.constants import ROOM_TEMP
from PySDM.backends.numba.storage import Storage
from PySDM.dynamics.aqueous_chemistry.aqueous_chemistry import equilibrate_pH
from PySDM.physics.constants import convert_to, ROOM_TEMP
from PySDM.physics import formulae as phys
from PySDM.physics import si
import numpy as np
import pytest
from collections import defaultdict


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
    @pytest.mark.parametrize('init_conc', (
            defaultdict(float, {'H2O': 1, 'NH3': 0.5, 'H2CO3(aq)': 0.05, 'H2SO3(aq)': 0.05}),  # TODO: realistic concentrations
    ))
    def test_equilibrate_pH_non_trivial(init_conc):
        from chempy import Equilibrium
        from chempy.equilibria import EqSystem
        from chempy.chemistry import Species

        # autop -> autoprotonation
        # prot -> protonation

        # constants from Table 4 in Kreidenweis et al 2003
        equilibria = {
            'water': Equilibrium({'H2O'}, {'H+', 'OH-'}, 10 ** -14),  # unit "molar" assumed
            'ammonia': Equilibrium({'NH4+'}, {'NH3', 'H+'}, 10 ** -9.24),  # same here
            # TODO
            # 'nitric': Equilibrium.from_string("HNO3(aq) = H+ + NO3-; 15.4"),
            'sulfonic_first': Equilibrium.from_string("H2SO3(aq) = H+ + HSO3-; 1.54*10**-2"),  # H2O*SO2
            'sulfonic_second': Equilibrium.from_string("HSO3- = H+ + SO3-2; 6.6*10**-8"),
            # TODO
            # 'sulphuric_first': Equilibrium.from_string("HSO4- = H+ + SO4-2; 1.2*10**-2"),
            'carbonic_first': Equilibrium.from_string("H2CO3(aq) = H+ + HCO3-; 4.3*10**-7"),
            'carbonic_second': Equilibrium.from_string("HCO3- = H+ + CO3-2; 4.68*10**-11")
        }
        substances = [Species.from_formula(f) for f in 'H2O OH- H+ NH3 NH4+ H2CO3(aq) HCO3- CO3-2 H2SO3(aq) HSO3- SO3-2'.split()]
        eqsys = EqSystem(equilibria.values(), substances)

        x, sol, sane = eqsys.root(init_conc)
        assert sol['success'] and sane

        H_idx = 2
        assert substances[H_idx].name == 'H+'
        expected_pH = -np.log10(x[H_idx])

        n_sd = 1
        particles = {
            # input  # TODO!
            'conc_N_mIII': Storage.from_ndarray(np.zeros(n_sd)),
            'conc_N_V': Storage.from_ndarray(np.zeros(n_sd)),
            'conc_C_IV': Storage.from_ndarray(np.full(n_sd, 0*init_conc['H2CO3(aq)'])),  # TODO: /litre -> /m3
            'conc_S_IV': Storage.from_ndarray(np.zeros(n_sd)),
            'conc_S_VI': Storage.from_ndarray(np.zeros(n_sd)),
            'volume': Storage.from_ndarray(np.ones(n_sd)),
            # output
            'moles_H': Storage.from_ndarray(np.zeros(n_sd))
        }
        equilibrate_pH(
            super_droplet_ids=(0,),
            particles=particles,
            env_T=ROOM_TEMP
        )
        convert_to(particles['volume'].data, si.litre)
        actual_pH = -np.log10(particles['moles_H'].data / particles['volume'].data)

        np.testing.assert_almost_equal(actual_pH, expected_pH)

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

