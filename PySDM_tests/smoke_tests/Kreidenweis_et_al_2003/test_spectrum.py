from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.physics import si


class TestSepctrum:
    @staticmethod
    def test_at_t_0():
        # Arrange
        settings = Settings(n_sd=100, dt=1 * si.s, n_substep=5)
        settings.t_max = 0
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Assert
        key = 'S_VI'
        assert (output[f'dm_{key}/dlog_10(dry diameter)'][0] > 0).any()