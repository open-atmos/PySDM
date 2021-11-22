# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from matplotlib import pyplot
from PySDM_examples.Shipway_and_Hill_2012 import Simulation, Settings
from PySDM.physics import si


# pylint: disable=redefined-outer-name
def test_few_steps(plot=False):
    # Arrange
    settings = Settings(n_sd_per_gridbox=50, dt=30 * si.s, dz=50 * si.m)
    simulation = Simulation(settings)

    # Act
    output = simulation.run(nt=100)

    # Plot
    def profile(var):
        return np.mean(output[var][:, -20:], axis=1)

    if plot:
        for var in ('RH', 'S_max', 'T', 'qv', 'p', 'ql',
                    'ripening rate', 'activating rate', 'deactivating rate'):
            pyplot.plot(profile(var), output['z'], linestyle='--', marker='o')
            pyplot.ylabel('Z [m]')
            pyplot.xlabel(var + ' [' + simulation.particulator.products[var].unit + ']')
            pyplot.grid()
            pyplot.show()

    # Assert
    assert min(profile('ql')) == 0
    assert .1 < max(profile('ql')) < 1
    # assert max(profile('ripening_rate')) > 0 # TODO #521
    assert max(profile('activating rate')) == 0
    # assert max(profile('deactivating_rate')) > 0 TODO #521
