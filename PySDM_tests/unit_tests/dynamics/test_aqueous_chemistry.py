from PySDM.dynamics.aqueous_chemistry.support import K_H2O, M, EQUILIBRIUM_CONST
from PySDM.backends.numba.storage import Storage
from PySDM.dynamics.aqueous_chemistry.aqueous_chemistry import equilibrate_pH
from PySDM.physics.constants import convert_to, ROOM_TEMP
from PySDM.physics import formulae as phys
from PySDM.physics import si

from chempy import Equilibrium
from chempy.equilibria import EqSystem
from chempy.chemistry import Species

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
        # TODO #434
        #  defaultdict(float, {'H2O': 1, 'HSO4-': 5e-3, 'HNO3(aq)': 0.035e-3, 'NH3': 5e-3, 'H2CO3(aq)': 0.01e-3, 'H2SO3(aq)': 0.005e-3}),
        defaultdict(float, {'H2O': 1, 'NH3': 5e-3, 'H2CO3(aq)': 0.01e-3, 'H2SO3(aq)': 0.005e-3}),
        defaultdict(float, {'H2O': 1, 'NH3': .5e-3, 'H2CO3(aq)': 0.1e-3, 'H2SO3(aq)': 0.05e-3}),
    ))
    @pytest.mark.parametrize('env_T', (ROOM_TEMP, ROOM_TEMP-30, ROOM_TEMP+30))
    def test_equilibrate_pH_non_trivial(init_conc, env_T):

        equilibria = {
            'water': Equilibrium.from_string(f"H2O = H+ + OH-; {K_H2O / M / M}"),
            'ammonia': Equilibrium.from_string(f"NH3 + H2O = NH4+ + OH-; {EQUILIBRIUM_CONST['K_NH3'].at(env_T) / M}"),
            # TODO #434
            # 'nitric': Equilibrium.from_string(f"HNO3(aq) = H+ + NO3-; {EQUILIBRIUM_CONST['K_HNO3'].at(env_T) / M}"),
            'sulfonic_first': Equilibrium.from_string(f"H2SO3(aq) = H+ + HSO3-; {EQUILIBRIUM_CONST['K_SO2'].at(env_T) / M}"),
            'sulfonic_second': Equilibrium.from_string(f"HSO3- = H+ + SO3-2; {EQUILIBRIUM_CONST['K_HSO3'].at(env_T) / M}"),
            # TODO #434
            # 'sulphuric_first': Equilibrium.from_string(f"HSO4- = H+ + SO4-2; {EQUILIBRIUM_CONST['K_HSO4'].at(env_T) / M}"),
            'carbonic_first': Equilibrium.from_string(f"H2CO3(aq) = H+ + HCO3-; {EQUILIBRIUM_CONST['K_CO2'].at(env_T) / M}"),
            'carbonic_second': Equilibrium.from_string(f"HCO3- = H+ + CO3-2; {EQUILIBRIUM_CONST['K_HCO3'].at(env_T) / M}")
        }
        substances = [
            Species.from_formula(f) for f in 'H2O OH- H+ NH3 NH4+ H2CO3(aq) HCO3- CO3-2 H2SO3(aq) HSO3- SO3-2'.split()
        ]
        eqsys = EqSystem(equilibria.values(), substances)

        x, sol, sane = eqsys.root(init_conc)
        assert sol['success'] and sane

        H_idx = 2
        assert substances[H_idx].name == 'H+'
        expected_pH = -np.log10(x[H_idx])

        n_sd = 1
        particles = {
            # input
            'conc_N_mIII': Storage.from_ndarray(np.full(n_sd, init_conc['NH3'] * 1e3)),
            'conc_N_V': Storage.from_ndarray(np.full(n_sd, init_conc['HNO3(aq)'] * 1e3)),
            'conc_C_IV': Storage.from_ndarray(np.full(n_sd, init_conc['H2CO3(aq)'] * 1e3)),
            'conc_S_IV': Storage.from_ndarray(np.full(n_sd, init_conc['H2SO3(aq)'] * 1e3)),
            'conc_S_VI': Storage.from_ndarray(np.full(n_sd, init_conc['HSO4-'] * 1e3)),
            'volume': Storage.from_ndarray(np.full(n_sd, phys.volume(radius=1*si.um))),
            # output
            'moles_H': Storage.from_ndarray(np.zeros(n_sd))
        }
        equilibrate_pH(
            super_droplet_ids=(0,),
            particles=particles,
            env_T=env_T
        )
        convert_to(particles['volume'].data, si.litre)
        actual_pH = -np.log10(particles['moles_H'].data / particles['volume'].data)

        np.testing.assert_allclose(actual_pH, expected_pH, rtol=1e-3)

    @staticmethod
    def test_oxidize():
        pass
        # TODO!

    @staticmethod
    def test_henry():
        from chempy.henry import Henry
        kH_O2 = Henry(1.2e-3, 1800, ref='carpenter_1966')

        from PySDM.dynamics.aqueous_chemistry.support import HENRY_CONST

