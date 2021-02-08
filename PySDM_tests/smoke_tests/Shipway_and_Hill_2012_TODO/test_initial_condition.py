from PySDM_examples.Shipway_and_Hill_2012 import Simulation, Settings
from PySDM.physics import si
import numpy as np


class TestInitialCondition:
    @staticmethod
    def test_initial_condition(plot=False):
        # Arrange
        settings = Settings(n_sd=100, w_1=1 * si.m / si.s)
        simulation = Simulation(settings)

        # Act
        output = simulation.run(nt=0)

        # Plot
        if plot:
            from matplotlib import pyplot
            for var in ('RH', 'T_ambient', 'qv', 'p_ambient'):
                pyplot.plot(output[var][:, 0], output['z'], linestyle='--', marker='o')
                pyplot.ylabel('Z [m]')
                pyplot.xlabel(var + ' [' + simulation.core.products[var].unit + ']')
                pyplot.grid()
                pyplot.show()

        # Assert
        assert output['RH'].shape == (settings.nz, 1)

        assert 25 < np.amin(output['RH']) < 30
        assert 90 < np.amax(output['RH']) < 100

        assert 740 * si.hPa < np.amin(output['p_ambient']) < 750 * si.hPa
        assert (np.diff(output['p_ambient']) < 0).all()
        assert 1000 * si.hPa < np.amax(output['p_ambient']) < 1050 * si.hPa

        assert 285 * si.K < np.amin(output['T_ambient']) < 286 * si.K
        assert output['T_ambient'][0] > np.amin(output['T_ambient'])
        assert 299 * si.K < np.amax(output['T_ambient']) < 300 * si.K
