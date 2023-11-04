# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from chempy import Substance
from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation

from PySDM.dynamics.impl.chemistry_utils import AQUEOUS_COMPOUNDS, SpecificGravities
from PySDM.physics import si
from PySDM.physics.constants import PPB, convert_to


class TestTable3:
    @staticmethod
    def test_at_t_0():
        # Arrange
        settings = Settings(n_sd=100, dt=1 * si.s, n_substep=5)
        settings.t_max = 0
        simulation = Simulation(settings)
        specific_gravities = SpecificGravities(
            simulation.particulator.formulae.constants
        )

        # Act
        output = simulation.run()

        # Assert
        np.testing.assert_allclose(output["RH"][0], 95)
        np.testing.assert_allclose(output["gas_S_IV_ppb"][0], 0.2)
        np.testing.assert_allclose(output["gas_N_mIII_ppb"][0], 0.1)
        np.testing.assert_allclose(output["gas_H2O2_ppb"], 0.5)
        np.testing.assert_allclose(output["gas_N_V_ppb"], 0.1)
        np.testing.assert_allclose(output["gas_O3_ppb"], 50)
        np.testing.assert_allclose(output["gas_C_IV_ppb"], 360 * 1000)

        rtol = 0.15

        mass_conc_SO4mm = 2
        mass_conc_NH4p = 0.375
        num_conc_SO4mm = (
            mass_conc_SO4mm / Substance.from_formula("SO4").mass * si.gram / si.mole
        )
        num_conc_NH4p = (
            mass_conc_NH4p / Substance.from_formula("NH4").mass * si.gram / si.mole
        )
        np.testing.assert_allclose(num_conc_NH4p, num_conc_SO4mm, rtol=0.005)
        mass_conc_H = (
            num_conc_NH4p * Substance.from_formula("H").mass * si.gram / si.mole
        )
        np.testing.assert_allclose(
            actual=np.asarray(output["q_dry"]) * np.asarray(output["rhod"]),
            desired=mass_conc_NH4p + mass_conc_SO4mm + mass_conc_H,
            rtol=rtol,
        )

        expected = {k: 0 for k in AQUEOUS_COMPOUNDS}
        expected["S_VI"] = mass_conc_SO4mm * si.ug / si.m**3
        expected["N_mIII"] = mass_conc_NH4p * si.ug / si.m**3

        for key in expected:
            mole_fraction = np.asarray(output[f"aq_{key}_ppb"])
            convert_to(mole_fraction, 1 / PPB)
            compound = AQUEOUS_COMPOUNDS[key][0]  # sic!
            np.testing.assert_allclose(
                actual=(
                    settings.formulae.trivia.mole_fraction_2_mixing_ratio(
                        mole_fraction, specific_gravity=specific_gravities[compound]
                    )
                    * np.asarray(output["rhod"])
                ),
                desired=expected[key],
                rtol=rtol,
            )

    @staticmethod
    def test_at_cloud_base():
        # Arrange
        settings = Settings(n_sd=50, dt=1 * si.s, n_substep=5)
        settings.t_max = 196 * si.s
        settings.output_interval = settings.dt
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Assert
        assert round(output["z"][-1]) == (698 - 600) * si.m
        np.testing.assert_allclose(output["p"][-1], 939 * si.mbar, rtol=0.005)
        np.testing.assert_allclose(output["T"][-1], 284.2 * si.K, rtol=0.005)
        np.testing.assert_allclose(
            settings.formulae.state_variable_triplet.rho_of_rhod_and_water_vapour_mixing_ratio(
                rhod=output["rhod"][-1],
                water_vapour_mixing_ratio=output["water vapour mixing ratio"][-1]
                * si.g
                / si.kg,
            ),
            1.15 * si.kg / si.m**3,
            rtol=0.005,
        )
        assert output["liquid water mixing ratio"][-2] < 0.00055
        assert output["liquid water mixing ratio"][-1] > 0.0004
        assert output["RH"][-1] > 100
        assert output["RH"][-8] < 100

    @staticmethod
    def test_at_1200m_above_cloud_base():
        # Arrange
        settings = Settings(n_sd=10, dt=1 * si.s, n_substep=5)
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Assert
        np.testing.assert_allclose(output["z"][-1], (1.2 + 0.1) * si.km, rtol=0.005)
        np.testing.assert_allclose(
            output["liquid water mixing ratio"][-1], 2.17, rtol=0.02
        )
