# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
import numpy as np
from chempy.electrolytes import ionic_strength
from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.backends.impl_numba.methods.chemistry_methods import calc_ionic_strength, _K, _conc
from PySDM.physics.constants import rho_w, ROOM_TEMP, K_H2O
from PySDM import Formulae
from PySDM.physics.aqueous_chemistry.support import EquilibriumConsts


@pytest.mark.parametrize("nt", (0, 1, 2, 3))
@pytest.mark.parametrize("n_sd", (10, 20))
def test_calc_ionic_strength(nt, n_sd):
    formulae = Formulae()
    EQUILIBRIUM_CONST = EquilibriumConsts(formulae).EQUILIBRIUM_CONST

    K = _K(
        NH3=EQUILIBRIUM_CONST["K_NH3"].at(ROOM_TEMP),
        SO2=EQUILIBRIUM_CONST["K_SO2"].at(ROOM_TEMP),
        HSO3=EQUILIBRIUM_CONST["K_HSO3"].at(ROOM_TEMP),
        HSO4=EQUILIBRIUM_CONST["K_HSO4"].at(ROOM_TEMP),
        HCO3=EQUILIBRIUM_CONST["K_HCO3"].at(ROOM_TEMP),
        CO2=EQUILIBRIUM_CONST["K_CO2"].at(ROOM_TEMP),
        HNO3=EQUILIBRIUM_CONST["K_HNO3"].at(ROOM_TEMP)
    )

    settings = Settings(dt=1, n_sd=n_sd, n_substep=5)
    settings.t_max = nt * settings.dt
    simulation = Simulation(settings)

    simulation.run()

    H = simulation.particulator.attributes['conc_N_mIII'].data
    conc = {
        'H+': H,
        'N-3': simulation.particulator.attributes['conc_N_mIII'].data,
        'N+5': simulation.particulator.attributes['conc_N_V'].data,
        'S+4': simulation.particulator.attributes['conc_S_IV'].data,
        'S+6': simulation.particulator.attributes['conc_S_VI'].data,
        'C+4': simulation.particulator.attributes['conc_C_IV'].data,
    }

    alpha_C = (1 + K.CO2 / conc['H+'] + K.CO2 * K.HCO3 / conc['H+'] ** 2)
    alpha_S = (1 + K.SO2 / conc['H+'] + K.SO2 * K.HSO3 / conc['H+'] ** 2)
    alpha_N3 = (1 + conc['H+'] * K.NH3 / K_H2O)
    alpha_N5 = (1 + K.HNO3 / conc['H+'])

    actual = calc_ionic_strength(
        H=conc['H+'],
        conc=_conc(
            N_mIII=conc['N-3'],
            N_V=conc['N+5'],
            C_IV=conc['C+4'],
            S_IV=conc['S+4'],
            S_VI=conc['S+6'],
        ),
        K=K
    )

    expected = ionic_strength({
        'H+': conc['H+'] / rho_w,
        'HCO3-': K.CO2 / conc['H+'] * conc['C+4'] / alpha_C / rho_w,
        'CO3-2': K.CO2 / conc['H+'] * K.HCO3 / conc['H+'] * conc['C+4'] / alpha_C / rho_w,
        'HSO3-': K.SO2 / conc['H+'] * conc['S+4'] / alpha_S / rho_w,
        'SO3-2': K.SO2 / conc['H+'] * K.HSO3 / conc['H+'] * conc['S+4'] / alpha_S / rho_w,
        'NH4+': K.NH3 / K_H2O * conc['H+'] * conc['N-3'] / alpha_N3 / rho_w,
        'NO3-': K.HNO3 / conc['H+'] * conc['N+5'] / alpha_N5 / rho_w,
        'HSO4-': conc['H+'] * conc['S+6'] / (conc['H+'] + K.HSO4) / rho_w,
        'SO4-2': K.HSO4 * conc['S+6'] / (conc['H+'] + K.HSO4) / rho_w,
        'OH-': K_H2O / conc['H+'] / rho_w
    }, warn=False) * rho_w

    np.testing.assert_allclose(actual, expected, rtol=1e-15)
