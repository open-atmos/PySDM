from PySDM_examples.Shipway_and_Hill_2012 import Simulation, Settings
from PySDM.physics import si


class TestFewTimesteps:
    @staticmethod
    def test_cloud_water_mixing_ratio(plot=False):
        # Arrange
        settings = Settings(n_sd_per_gridbox=5, dt=10*si.s, dz=100*si.m)
        simulation = Simulation(settings)

        # Act
        output = simulation.run(nt=5)

        # Plot
        if plot:
            from matplotlib import pyplot
            for var in ('RH', 'T_ambient', 'qv', 'p_ambient', 'ql'):
                pyplot.plot(output[var][:, -1], output['z'], linestyle='--', marker='o')
                pyplot.ylabel('Z [m]')
                pyplot.xlabel(var + ' [' + simulation.core.products[var].unit + ']')
                pyplot.grid()
                pyplot.show()

        # Assert
        ql = output['ql'][:, -1]
        assert min(ql) == 0
        assert .3 < max(ql) < .6
