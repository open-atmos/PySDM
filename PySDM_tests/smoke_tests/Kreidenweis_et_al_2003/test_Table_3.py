from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.physics import si
from PySDM.physics import formulae as phys
import numpy as np


class TestTable3:
    @staticmethod
    def test_at_t_0():
        # Arrange
        settings = Settings(n_sd=1, dt=1 * si.s)
        simulation = Simulation(settings)
        zero = 0

        # Act
        output = simulation.run(nt=zero)

        # Assert
        np.testing.assert_allclose(output['RH_env'][zero], 95)
        np.testing.assert_allclose(output['gas_SO2_ppb'][zero], 0.2)
        np.testing.assert_allclose(output['gas_NH3_ppb'][zero], 0.1)
        np.testing.assert_allclose(output['gas_H2O2_ppb'], 0.5)
        np.testing.assert_allclose(output['gas_HNO3_ppb'], 0.1)
        np.testing.assert_allclose(output['gas_O3_ppb'], 50)
        np.testing.assert_allclose(output['gas_CO2_ppb'], 360*1000)

        # TODO
        # SO4= (particulate) at t = 0 	2 (μg m−3)
        # NH4+ (particulate) at t = 0 	0.375 (μg m−3)

    @staticmethod
    def test_at_cloud_base():
        # Arrange
        settings = Settings(n_sd=100, dt=1*si.s)
        simulation = Simulation(settings)

        # Act
        output = simulation.run(nt=int(196 * si.s / settings.dt))

        # Assert
        assert round(output['z'][-1]) == (698 - 600) * si.m
        np.testing.assert_allclose(output['p_env'][-1], 939 * si.mbar, rtol=.005)
        np.testing.assert_allclose(output['T_env'][-1], 284.2 * si.K, rtol=.005)
        np.testing.assert_allclose(
            phys.MoistAir.rho_of_rhod_qv(rhod=output['rhod_env'][-1], qv=output['qv_env'][-1]*si.g/si.kg),
            1.15 * si.kg / si.m**3,
            rtol=.005
        )
        assert output['ql'][-2] < .0005
        assert output['ql'][-1] > .0004
        assert output['RH_env'][-1] > 100
        assert output['RH_env'][-6] < 100

    def test_at_1200m_above_cloud_base(self):
        # Arrange
        settings = Settings(n_sd=100, dt=4 * si.s)
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Assert
        np.testing.assert_allclose(output['z'][-1], (1.2 + .1) * si.km, rtol=.005)
        np.testing.assert_allclose(output['ql'][-1], 2.17, rtol=.02)

