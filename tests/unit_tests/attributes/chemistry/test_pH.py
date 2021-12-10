from PySDM.physics.aqueous_chemistry.support import M, EquilibriumConsts
from PySDM.physics.constants import ROOM_TEMP, K_H2O
from PySDM.physics.formulae import Formulae
from PySDM.backends.numba.impl._chemistry_methods import ChemistryMethods
from PySDM.dynamics import aqueous_chemistry
from chempy import Equilibrium
from chempy.equilibria import EqSystem
from chempy.chemistry import Species

import numpy as np
import pytest
from collections import defaultdict


formulae = Formulae()
EQUILIBRIUM_CONST = EquilibriumConsts(formulae).EQUILIBRIUM_CONST


class Test_pH:
    @staticmethod
    def test_equilibrate_pH_pure_water():
        # Arrange
        eqs = {}
        for key in EQUILIBRIUM_CONST.keys():
            eqs[key] = np.full(1, EQUILIBRIUM_CONST[key].at(ROOM_TEMP))

        # Act
        result = np.empty(1)
        ChemistryMethods.equilibrate_H_body(
            within_tolerance=formulae.trivia.within_tolerance,
            pH2H=formulae.trivia.pH2H,
            H2pH=formulae.trivia.H2pH,
            N_mIII=np.zeros(1),
            N_V=np.zeros(1),
            C_IV=np.zeros(1),
            S_IV=np.zeros(1),
            S_VI=np.zeros(1),
            K_HNO3=eqs['K_HNO3'].data,
            K_HCO3=eqs['K_HCO3'].data,
            K_HSO3=eqs['K_HSO3'].data,
            K_HSO4=eqs['K_HSO4'].data,
            K_CO2=eqs['K_CO2'].data,
            K_NH3=eqs['K_NH3'].data,
            K_SO2=eqs['K_SO2'].data,
            cell_id=np.zeros(1, dtype=int),
            # output
            do_chemistry_flag=np.empty(1),
            pH=result,
            # params
            H_min=formulae.trivia.pH2H(aqueous_chemistry.default_pH_max),
            H_max=formulae.trivia.pH2H(aqueous_chemistry.default_pH_min),
            ionic_strength_threshold=aqueous_chemistry.default_ionic_strength_threshold,
            rtol=aqueous_chemistry.default_pH_rtol
        )

        # Assert
        np.testing.assert_allclose(result, 7)

    @staticmethod
    @pytest.mark.parametrize('init_conc', (
        defaultdict(float, {'H2O': 1, 'NH3': 5e-3, 'H2CO3(aq)': 0.01e-3, 'H2SO3(aq)': 0.005e-3}),
        defaultdict(float, {'H2O': 1, 'NH3': .5e-3, 'H2CO3(aq)': 0.1e-3, 'H2SO3(aq)': 0.05e-3}),
    ))
    @pytest.mark.parametrize('env_T', (ROOM_TEMP, ROOM_TEMP-30, ROOM_TEMP+30))
    def test_equilibrate_pH_non_trivial(init_conc, env_T):

        equilibria = {
            'water': Equilibrium.from_string(f"H2O = H+ + OH-; {K_H2O / M / M}"),
            'ammonia': Equilibrium.from_string(f"NH3 + H2O = NH4+ + OH-; {EQUILIBRIUM_CONST['K_NH3'].at(env_T) / M}"),
            'sulfonic_first': Equilibrium.from_string(f"H2SO3(aq) = H+ + HSO3-; {EQUILIBRIUM_CONST['K_SO2'].at(env_T) / M}"),
            'sulfonic_second': Equilibrium.from_string(f"HSO3- = H+ + SO3-2; {EQUILIBRIUM_CONST['K_HSO3'].at(env_T) / M}"),
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

        actual_pH = np.empty(1)
        formulae = Formulae()
        ChemistryMethods.equilibrate_H_body(
            within_tolerance=formulae.trivia.within_tolerance,
            pH2H=formulae.trivia.pH2H,
            H2pH=formulae.trivia.H2pH,
            N_mIII=np.full(1, init_conc['NH3'] * 1e3),
            N_V=np.full(1, init_conc['HNO3(aq)'] * 1e3),
            C_IV=np.full(1, init_conc['H2CO3(aq)'] * 1e3),
            S_IV=np.full(1, init_conc['H2SO3(aq)'] * 1e3),
            S_VI=np.full(1, init_conc['HSO4-'] * 1e3),
            K_HNO3=eqs['K_HNO3'].data,
            K_HCO3=eqs['K_HCO3'].data,
            K_HSO3=eqs['K_HSO3'].data,
            K_HSO4=eqs['K_HSO4'].data,
            K_CO2=eqs['K_CO2'].data,
            K_NH3=eqs['K_NH3'].data,
            K_SO2=eqs['K_SO2'].data,
            cell_id=np.zeros(1, dtype=int),
            # output
            do_chemistry_flag=np.empty(1),
            pH=actual_pH,
            # params
            H_min=formulae.trivia.pH2H(aqueous_chemistry.default_pH_max),
            H_max=formulae.trivia.pH2H(aqueous_chemistry.default_pH_min),
            ionic_strength_threshold=aqueous_chemistry.default_ionic_strength_threshold,
            rtol=aqueous_chemistry.default_pH_rtol
        )

        np.testing.assert_allclose(actual_pH[0], expected_pH, rtol=1e-5)
