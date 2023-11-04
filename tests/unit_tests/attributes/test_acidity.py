# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from collections import defaultdict

import numpy as np
import pytest
from chempy import Equilibrium
from chempy.chemistry import Species
from chempy.equilibria import EqSystem

from PySDM.backends.impl_numba.methods.chemistry_methods import (
    _K,
    ChemistryMethods,
    _conc,
)
from PySDM.dynamics import aqueous_chemistry
from PySDM.dynamics.impl.chemistry_utils import EquilibriumConsts, M
from PySDM.formulae import Formulae
from PySDM.physics.constants import K_H2O

FORMULAE = Formulae()
EQUILIBRIUM_CONST = EquilibriumConsts(FORMULAE).EQUILIBRIUM_CONST


class TestAcidity:
    @staticmethod
    def test_equilibrate_pH_pure_water():
        # Arrange
        eqs = {}
        for key, const in EQUILIBRIUM_CONST.items():
            eqs[key] = np.full(1, const.at(FORMULAE.constants.ROOM_TEMP))

        # Act
        result = np.empty(1)
        ChemistryMethods.equilibrate_H_body(
            within_tolerance=FORMULAE.trivia.within_tolerance,
            pH2H=FORMULAE.trivia.pH2H,
            H2pH=FORMULAE.trivia.H2pH,
            conc=_conc(
                N_mIII=np.zeros(1),
                N_V=np.zeros(1),
                C_IV=np.zeros(1),
                S_IV=np.zeros(1),
                S_VI=np.zeros(1),
            ),
            K=_K(
                HNO3=eqs["K_HNO3"].data,
                HCO3=eqs["K_HCO3"].data,
                HSO3=eqs["K_HSO3"].data,
                HSO4=eqs["K_HSO4"].data,
                CO2=eqs["K_CO2"].data,
                NH3=eqs["K_NH3"].data,
                SO2=eqs["K_SO2"].data,
            ),
            cell_id=np.zeros(1, dtype=int),
            # output
            do_chemistry_flag=np.empty(1),
            pH=result,
            # params
            H_min=FORMULAE.trivia.pH2H(aqueous_chemistry.DEFAULTS.pH_max),
            H_max=FORMULAE.trivia.pH2H(aqueous_chemistry.DEFAULTS.pH_min),
            ionic_strength_threshold=aqueous_chemistry.DEFAULTS.ionic_strength_threshold,
            rtol=aqueous_chemistry.DEFAULTS.pH_rtol,
        )

        # Assert
        np.testing.assert_allclose(result, 7)

    @staticmethod
    @pytest.mark.parametrize(
        "init_conc",
        (
            defaultdict(
                float,
                {"H2O": 1, "NH3": 5e-3, "H2CO3(aq)": 0.01e-3, "H2SO3(aq)": 0.005e-3},
            ),
            defaultdict(
                float,
                {"H2O": 1, "NH3": 0.5e-3, "H2CO3(aq)": 0.1e-3, "H2SO3(aq)": 0.05e-3},
            ),
        ),
    )
    @pytest.mark.parametrize(
        "env_T",
        (
            FORMULAE.constants.ROOM_TEMP,
            FORMULAE.constants.ROOM_TEMP - 30,
            FORMULAE.constants.ROOM_TEMP + 30,
        ),
    )
    def test_equilibrate_pH_non_trivial(init_conc, env_T):
        equilibria = {
            "water": Equilibrium.from_string(f"H2O = H+ + OH-; {K_H2O / M / M}"),
            "ammonia": Equilibrium.from_string(
                f"NH3 + H2O = NH4+ + OH-; {EQUILIBRIUM_CONST['K_NH3'].at(env_T) / M}"
            ),
            "sulfonic_first": Equilibrium.from_string(
                f"H2SO3(aq) = H+ + HSO3-; {EQUILIBRIUM_CONST['K_SO2'].at(env_T) / M}"
            ),
            "sulfonic_second": Equilibrium.from_string(
                f"HSO3- = H+ + SO3-2; {EQUILIBRIUM_CONST['K_HSO3'].at(env_T) / M}"
            ),
            "carbonic_first": Equilibrium.from_string(
                f"H2CO3(aq) = H+ + HCO3-; {EQUILIBRIUM_CONST['K_CO2'].at(env_T) / M}"
            ),
            "carbonic_second": Equilibrium.from_string(
                f"HCO3- = H+ + CO3-2; {EQUILIBRIUM_CONST['K_HCO3'].at(env_T) / M}"
            ),
        }
        substances = [
            Species.from_formula(f)
            for f in "H2O OH- H+ NH3 NH4+ H2CO3(aq) HCO3- CO3-2 H2SO3(aq) HSO3- SO3-2".split()
        ]
        eqsys = EqSystem(equilibria.values(), substances)

        x, sol, sane = eqsys.root(init_conc)
        assert sol["success"] and sane

        H_idx = 2
        assert substances[H_idx].name == "H+"
        expected_pH = -np.log10(x[H_idx])

        eqs = {}
        for key, const in EQUILIBRIUM_CONST.items():
            eqs[key] = np.full(1, const.at(env_T))

        actual_pH = np.empty(1)
        ChemistryMethods.equilibrate_H_body(
            within_tolerance=FORMULAE.trivia.within_tolerance,
            pH2H=FORMULAE.trivia.pH2H,
            H2pH=FORMULAE.trivia.H2pH,
            conc=_conc(
                N_mIII=np.full(1, init_conc["NH3"] * 1e3),
                N_V=np.full(1, init_conc["HNO3(aq)"] * 1e3),
                C_IV=np.full(1, init_conc["H2CO3(aq)"] * 1e3),
                S_IV=np.full(1, init_conc["H2SO3(aq)"] * 1e3),
                S_VI=np.full(1, init_conc["HSO4-"] * 1e3),
            ),
            K=_K(
                HNO3=eqs["K_HNO3"].data,
                HCO3=eqs["K_HCO3"].data,
                HSO3=eqs["K_HSO3"].data,
                HSO4=eqs["K_HSO4"].data,
                CO2=eqs["K_CO2"].data,
                NH3=eqs["K_NH3"].data,
                SO2=eqs["K_SO2"].data,
            ),
            cell_id=np.zeros(1, dtype=int),
            # output
            do_chemistry_flag=np.empty(1),
            pH=actual_pH,
            # params
            H_min=FORMULAE.trivia.pH2H(aqueous_chemistry.DEFAULTS.pH_max),
            H_max=FORMULAE.trivia.pH2H(aqueous_chemistry.DEFAULTS.pH_min),
            ionic_strength_threshold=aqueous_chemistry.DEFAULTS.ionic_strength_threshold,
            rtol=aqueous_chemistry.DEFAULTS.pH_rtol,
        )

        np.testing.assert_allclose(actual_pH[0], expected_pH, rtol=1e-5)
