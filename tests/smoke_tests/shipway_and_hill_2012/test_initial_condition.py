# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from matplotlib import pyplot
from PySDM_examples.Shipway_and_Hill_2012 import Simulation, Settings
from PySDM.physics import si


class TestInitialCondition:
    @staticmethod
    def test_initial_condition(plot=False):
        # Arrange
        settings = Settings(n_sd_per_gridbox=100, rho_times_w_1=1 * si.m / si.s)
        simulation = Simulation(settings)

        # Act
        output = simulation.run(nt=0)

        # Plot
        if plot:
            for var in ('RH', 'T', 'qv', 'p'):
                pyplot.plot(output[var][:, 0], output['z'], linestyle='--', marker='o')
                pyplot.ylabel('Z [m]')
                pyplot.xlabel(var + ' [' + simulation.particulator.products[var].unit + ']')
                pyplot.grid()
                pyplot.show()

        # Assert
        assert output['RH'].shape == (settings.nz, 1)

        assert 35 < np.amin(output['RH']) < 40
        assert 110 < np.amax(output['RH']) < 115

        assert 700 * si.hPa < np.amin(output['p']) < 710 * si.hPa
        assert (np.diff(output['p']) < 0).all()
        assert 950 * si.hPa < np.amax(output['p']) < 1000 * si.hPa

        assert 280 * si.K < np.amin(output['T']) < 285 * si.K
        assert output['T'][0] > np.amin(output['T'])
        assert 295 * si.K < np.amax(output['T']) < 300 * si.K
