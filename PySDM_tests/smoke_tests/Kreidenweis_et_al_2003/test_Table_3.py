from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.physics import si
from PySDM.physics import formulae as phys
from PySDM.physics import constants as const
import numpy as np


class TestTable3:
    @staticmethod
    def test_at_t_0():
        # Arrange
        settings = Settings(n_sd=1, dt=1 * si.s)
        simulation = Simulation(settings)

        # Act
        output = simulation.run(nt=0)

        # Assert
        np.testing.assert_allclose(output['RH'][0], 95)
        # TODO
        # SO2 at t = 0 	200 (ppt‐v)
        np.testing.assert_allclose(output['SO2_tot_conc'], 0.2 * const.ppb)
        # NH3(g) at t = 0 	100 (ppt‐v)
        # H2O2 at t = 0 	500 (ppt‐v)
        # HNO3 at t = 0 	100 (ppt‐v)
        # O3 at t = 0 	50 (ppb‐v)
        # CO2 at t = 0 	360 (ppm‐v)
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
        np.testing.assert_allclose(output['p_ambient'][-1], 939 * si.mbar, rtol=.005)
        np.testing.assert_allclose(output['T_ambient'][-1], 284.2 * si.K, rtol=.005)
        np.testing.assert_allclose(
            phys.MoistAir.rho_of_rhod_qv(rhod=output['rhod'][-1], qv=output['qv'][-1]*si.g/si.kg),
            1.15 * si.kg / si.m**3,
            rtol=.005
        )
        assert output['ql'][-2] < .0005
        assert output['ql'][-1] > .0004
        assert output['RH'][-1] > 100
        assert output['RH'][-6] < 100

    def test_at_1200m_above_cloud_base(self):
        # Arrange
        settings = Settings(n_sd=100, dt=2 * si.s)
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Assert
        np.testing.assert_allclose(output['z'][-1], (1.2 + .1) * si.km, rtol=.005)
        np.testing.assert_allclose(output['ql'][-1], 2.17, rtol=.006)

