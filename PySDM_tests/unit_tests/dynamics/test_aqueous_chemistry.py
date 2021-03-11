from PySDM.physics.constants import ROOM_TEMP
from PySDM.backends.numba.storage import Storage
from PySDM.dynamics.aqueous_chemistry.aqueous_chemistry import equilibrate_pH, EQUILIBRIUM_CONST
from PySDM.physics.constants import convert_to
from PySDM.physics import formulae as phys
from PySDM.physics import si
import numpy as np
import pytest


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

    @staticmethod
    @pytest.mark.parametrize("nt", (0, 1, 2, 3))
    @pytest.mark.parametrize("n_sd", (1, 2, 100))
    def test_calc_ionic_strength(nt, n_sd):
        from chempy.electrolytes import ionic_strength
        from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
        from PySDM.dynamics.aqueous_chemistry.aqueous_chemistry import calc_ionic_strength
        from PySDM.physics.constants import rho_w, ROOM_TEMP


        K_NH3 = EQUILIBRIUM_CONST["K_NH3"].at(ROOM_TEMP)
        K_SO2 = EQUILIBRIUM_CONST["K_SO2"].at(ROOM_TEMP)
        K_HSO3 = EQUILIBRIUM_CONST["K_HSO3"].at(ROOM_TEMP)
        K_HSO4 = EQUILIBRIUM_CONST["K_HSO4"].at(ROOM_TEMP)
        K_HCO3 = EQUILIBRIUM_CONST["K_HCO3"].at(ROOM_TEMP)
        K_CO2 = EQUILIBRIUM_CONST["K_CO2"].at(ROOM_TEMP)
        K_HNO3 = EQUILIBRIUM_CONST["K_HNO3"].at(ROOM_TEMP)
        K_H2O = EQUILIBRIUM_CONST["K_H2O"].at(ROOM_TEMP)

        settings = Settings(dt=1, n_sd=n_sd)
        simulation = Simulation(settings)
        simulation.run(nt)

        conc = {
            'H+': simulation.core.particles['conc_H'].data,
            'N-3': simulation.core.particles['conc_N_mIII'].data,
            'N+5': simulation.core.particles['conc_N_V'].data,
            'S+4': simulation.core.particles['conc_S_IV'].data,
            'S+6': simulation.core.particles['conc_S_VI'].data,
            'C+4': simulation.core.particles['conc_C_IV'].data,
        }

        alpha_C = (1 + K_CO2 / conc['H+'] + K_CO2 * K_HCO3 / conc['H+']**2)
        alpha_S = (1 + K_SO2 / conc['H+'] + K_SO2 * K_HSO3 / conc['H+']**2)
        alpha_N3 = (1 + conc['H+'] * K_NH3 / K_H2O)
        alpha_N5 = (1 + K_HNO3 / conc['H+'])


        actual = calc_ionic_strength(
            Hp=conc['H+'],
            N_III=conc['N-3'],
            N_V=conc['N+5'],
            C_IV=conc['C+4'],
            S_IV=conc['S+4'],
            S_VI=conc['S+6'],
            env_T=ROOM_TEMP
        )
        expected = ionic_strength({
            'H+': conc['H+'] / rho_w,
            'HCO3-': K_CO2 / conc['H+'] * conc['C+4'] / alpha_C / rho_w,
            'CO32-': K_CO2 / conc['H+'] * K_HCO3 / conc['H+'] * conc['C+4'] / alpha_C / rho_w,
            'HSO3-': K_SO2 / conc['H+'] * conc['S+4'] / alpha_S / rho_w,
            'SO32-': K_SO2 / conc['H+'] * K_HSO3 / conc['H+'] * conc['S+4'] / alpha_S / rho_w,
            'NH4+': K_NH3 / K_H2O * conc['H+'] * conc['N-3'] / alpha_N3 / rho_w,
            'NO3-': K_HNO3 / conc['H+'] * conc['N+5'] / alpha_N5 / rho_w,
            'HSO4-': conc['H+'] * conc['S+6'] / (conc['H+'] + K_HSO4) / rho_w,
            'SO42-': K_HSO4 * conc['S+6'] / (conc['H+'] + K_HSO4) / rho_w,
            'OH-': K_H2O / conc['H+'] / rho_w
        }) * rho_w

        np.testing.assert_allclose(actual, expected, rtol=1e-2)

