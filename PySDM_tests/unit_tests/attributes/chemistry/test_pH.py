from PySDM.dynamics.aqueous_chemistry.support import M, EQUILIBRIUM_CONST
from PySDM.physics.constants import ROOM_TEMP, K_H2O
from PySDM.attributes.chemistry.pH import equilibrate_H
from chempy import Equilibrium
from chempy.equilibria import EqSystem
from chempy.chemistry import Species

import numpy as np
import pytest
from collections import defaultdict


class Test_pH:
    @staticmethod
    def test_equilibrate_pH_pure_water():
        # Arrange
        eqs = {}
        for key in EQUILIBRIUM_CONST.keys():
            eqs[key] = np.full(1, EQUILIBRIUM_CONST[key].at(ROOM_TEMP))

        # Act
        result, _ = equilibrate_H(
            equilibrium_consts=eqs,
            cell_id=0,
            N_mIII=0,
            N_V=0,
            C_IV=0,
            S_IV=0,
            S_VI=0
        )

        # Assert
        np.testing.assert_allclose(result, 7)

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

        eqs = {}
        for key in EQUILIBRIUM_CONST.keys():
            eqs[key] = np.full(1, EQUILIBRIUM_CONST[key].at(env_T))

        actual_pH, _ = equilibrate_H(
            N_mIII=init_conc['NH3'] * 1e3,
            N_V=init_conc['HNO3(aq)'] * 1e3,
            C_IV=init_conc['H2CO3(aq)'] * 1e3,
            S_IV=init_conc['H2SO3(aq)'] * 1e3,
            S_VI=init_conc['HSO4-'] * 1e3,
            equilibrium_consts=eqs,
            cell_id=0
        )

        np.testing.assert_allclose(actual_pH, expected_pH, rtol=1e-5)

    @staticmethod
    @pytest.mark.parametrize("nt", (0, 1, 2, 3))
    @pytest.mark.parametrize("n_sd", (1, 2, 100))
    def test_calc_ionic_strength(nt, n_sd):
        from chempy.electrolytes import ionic_strength
        from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
        from PySDM.backends.numba.impl._chemistry_methods import calc_ionic_strength
        from PySDM.physics.constants import rho_w, ROOM_TEMP

        K_NH3 = EQUILIBRIUM_CONST["K_NH3"].at(ROOM_TEMP)
        K_SO2 = EQUILIBRIUM_CONST["K_SO2"].at(ROOM_TEMP)
        K_HSO3 = EQUILIBRIUM_CONST["K_HSO3"].at(ROOM_TEMP)
        K_HSO4 = EQUILIBRIUM_CONST["K_HSO4"].at(ROOM_TEMP)
        K_HCO3 = EQUILIBRIUM_CONST["K_HCO3"].at(ROOM_TEMP)
        K_CO2 = EQUILIBRIUM_CONST["K_CO2"].at(ROOM_TEMP)
        K_HNO3 = EQUILIBRIUM_CONST["K_HNO3"].at(ROOM_TEMP)

        settings = Settings(dt=1, n_sd=n_sd, n_substep=5)
        settings.t_max = nt * settings.dt
        simulation = Simulation(settings)
        simulation.run()

        H = simulation.core.particles['conc_N_mIII'].data
        conc = {
            'H+': H,
            'N-3': simulation.core.particles['conc_N_mIII'].data,
            'N+5': simulation.core.particles['conc_N_V'].data,
            'S+4': simulation.core.particles['conc_S_IV'].data,
            'S+6': simulation.core.particles['conc_S_VI'].data,
            'C+4': simulation.core.particles['conc_C_IV'].data,
        }

        alpha_C = (1 + K_CO2 / conc['H+'] + K_CO2 * K_HCO3 / conc['H+'] ** 2)
        alpha_S = (1 + K_SO2 / conc['H+'] + K_SO2 * K_HSO3 / conc['H+'] ** 2)
        alpha_N3 = (1 + conc['H+'] * K_NH3 / K_H2O)
        alpha_N5 = (1 + K_HNO3 / conc['H+'])

        actual = calc_ionic_strength(
            H=conc['H+'],
            N_mIII=conc['N-3'],
            N_V=conc['N+5'],
            C_IV=conc['C+4'],
            S_IV=conc['S+4'],
            S_VI=conc['S+6'],
            K_NH3=K_NH3,
            K_SO2=K_SO2,
            K_HSO3=K_HSO3,
            K_HSO4=K_HSO4,
            K_HCO3=K_HCO3,
            K_CO2=K_CO2,
            K_HNO3=K_HNO3
        )

        expected = ionic_strength({
            'H+': conc['H+'] / rho_w,
            'HCO3-': K_CO2 / conc['H+'] * conc['C+4'] / alpha_C / rho_w,
            'CO3-2': K_CO2 / conc['H+'] * K_HCO3 / conc['H+'] * conc['C+4'] / alpha_C / rho_w,
            'HSO3-': K_SO2 / conc['H+'] * conc['S+4'] / alpha_S / rho_w,
            'SO3-2': K_SO2 / conc['H+'] * K_HSO3 / conc['H+'] * conc['S+4'] / alpha_S / rho_w,
            'NH4+': K_NH3 / K_H2O * conc['H+'] * conc['N-3'] / alpha_N3 / rho_w,
            'NO3-': K_HNO3 / conc['H+'] * conc['N+5'] / alpha_N5 / rho_w,
            'HSO4-': conc['H+'] * conc['S+6'] / (conc['H+'] + K_HSO4) / rho_w,
            'SO4-2': K_HSO4 * conc['S+6'] / (conc['H+'] + K_HSO4) / rho_w,
            'OH-': K_H2O / conc['H+'] / rho_w
        }, warn=False) * rho_w  # TODO: warn=True if equilibrate_pH done

        np.testing.assert_allclose(actual, expected, rtol=1e-15)


