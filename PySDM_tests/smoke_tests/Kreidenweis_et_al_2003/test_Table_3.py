from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.physics import si
from PySDM.physics.constants import convert_to, ppb
from PySDM.physics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS, SPECIFIC_GRAVITY
import numpy as np
from chempy import Substance


class TestTable3:
    @staticmethod
    def test_at_t_0():
        # Arrange
        settings = Settings(n_sd=100, dt=1 * si.s, n_substep=5)
        settings.t_max = 0
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Assert
        np.testing.assert_allclose(output['RH_env'][0], 95)
        np.testing.assert_allclose(output['gas_S_IV_ppb'][0], 0.2)
        np.testing.assert_allclose(output['gas_N_mIII_ppb'][0], 0.1)
        np.testing.assert_allclose(output['gas_H2O2_ppb'], 0.5)
        np.testing.assert_allclose(output['gas_N_V_ppb'], 0.1)
        np.testing.assert_allclose(output['gas_O3_ppb'], 50)
        np.testing.assert_allclose(output['gas_C_IV_ppb'], 360*1000)

        rtol = 0.15

        mass_conc_SO4mm = 2
        mass_conc_NH4p = 0.375
        num_conc_SO4mm = mass_conc_SO4mm / Substance.from_formula("SO4").mass * si.gram / si.mole
        num_conc_NH4p = mass_conc_NH4p / Substance.from_formula("NH4").mass * si.gram / si.mole
        np.testing.assert_allclose(num_conc_NH4p, num_conc_SO4mm, rtol=.005)
        mass_conc_H = num_conc_NH4p * Substance.from_formula("H").mass * si.gram / si.mole
        np.testing.assert_allclose(
            actual=np.asarray(output['q_dry'])*np.asarray(output['rhod_env']),
            desired=mass_conc_NH4p + mass_conc_SO4mm + mass_conc_H,
            rtol=rtol
        )

        expected = {k: 0 for k in AQUEOUS_COMPOUNDS.keys()}
        expected['S_VI'] = mass_conc_SO4mm * si.ug / si.m**3
        expected['N_mIII'] = mass_conc_NH4p * si.ug / si.m**3

        for key in expected.keys():
            mole_fraction = np.asarray(output[f"aq_{key}_ppb"])
            convert_to(mole_fraction, 1/ppb)
            compound = AQUEOUS_COMPOUNDS[key][0]  # sic!
            np.testing.assert_allclose(
                actual=(
                    settings.formulae.trivia.mole_fraction_2_mixing_ratio(mole_fraction, specific_gravity=SPECIFIC_GRAVITY[compound])
                    * np.asarray(output['rhod_env'])
                ),
                desired=expected[key],
                rtol=rtol
            )

    @staticmethod
    def test_at_cloud_base():
        # Arrange
        settings = Settings(n_sd=50, dt=1*si.s, n_substep=5)
        settings.t_max = 196 * si.s
        settings.output_interval = settings.dt
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Assert
        assert round(output['z'][-1]) == (698 - 600) * si.m
        np.testing.assert_allclose(output['p_env'][-1], 939 * si.mbar, rtol=.005)
        np.testing.assert_allclose(output['T_env'][-1], 284.2 * si.K, rtol=.005)
        np.testing.assert_allclose(
            settings.formulae.state_variable_triplet.rho_of_rhod_qv(rhod=output['rhod_env'][-1], qv=output['qv_env'][-1]*si.g/si.kg),
            1.15 * si.kg / si.m**3,
            rtol=.005
        )
        assert output['ql'][-2] < .00055
        assert output['ql'][-1] > .0004
        assert output['RH_env'][-1] > 100
        assert output['RH_env'][-8] < 100

    def test_at_1200m_above_cloud_base(self):
        # Arrange
        settings = Settings(n_sd=10, dt=1 * si.s, n_substep=5)
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Assert
        np.testing.assert_allclose(output['z'][-1], (1.2 + .1) * si.km, rtol=.005)
        np.testing.assert_allclose(output['ql'][-1], 2.17, rtol=.02)
