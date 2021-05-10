from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.physics import si
from PySDM.initialisation.spectral_sampling import ConstantMultiplicity, Logarithmic
import pytest


class TestSepctrum:
    @staticmethod
    @pytest.mark.parametrize("spectral_sampling", [ConstantMultiplicity, Logarithmic])
    def test_at_t_0(spectral_sampling, plot=False):
        # Arrange
        settings = Settings(n_sd=64, dt=1 * si.s, n_substep=5, spectral_sampling=spectral_sampling)
        settings.t_max = 0
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Plot
        if plot:
            from matplotlib import pyplot
            pyplot.step(
                2e6 * settings.dry_radius_bins_edges[:-1],
                output['dm_S_VI/dlog_10(dry diameter)'][-1]
            )
            pyplot.ylabel('dS(VI)/dlog_10(D)')
            pyplot.xlabel('dry diameter [µm]')
            pyplot.xscale('log')
            pyplot.yscale('log')
            pyplot.ylim([.01, 12])
            pyplot.show()

        # Assert
        key = 'S_VI'
        # TODO #481 : better than >0 (we do have analytic formula)
        assert (output[f'dm_{key}/dlog_10(dry diameter)'][0] > 0).any()