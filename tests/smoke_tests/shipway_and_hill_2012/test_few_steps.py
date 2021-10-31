import numpy as np
from PySDM.physics import si
from PySDM_examples.Shipway_and_Hill_2012 import Simulation, Settings
# noinspection PyUnresolvedReferences
from ...backends_fixture import backend


def test_few_steps(backend, plot=False):
    # Arrange
    settings = Settings(n_sd_per_gridbox=50, dt=30 * si.s, dz=50 * si.m)
    simulation = Simulation(settings)

    # Act
    output = simulation.run(nt=100)

    # Plot
    def profile(var):
        return np.mean(output[var][:, -20:], axis=1)

    if plot:
        from matplotlib import pyplot
        for var in ('RH_env', 'S_max', 'T_env', 'qv_env', 'p_env', 'ql', 'ripening_rate', 'activating_rate', 'deactivating_rate'):
            pyplot.plot(profile(var), output['z'], linestyle='--', marker='o')
            pyplot.ylabel('Z [m]')
            pyplot.xlabel(var + ' [' + simulation.particulator.products[var].unit + ']')
            pyplot.grid()
            pyplot.show()

    # Assert
    assert min(profile('ql')) == 0
    assert .1 < max(profile('ql')) < 1
    # assert max(profile('ripening_rate')) > 0 # TODO #521
    assert max(profile('activating_rate')) == 0
    # assert max(profile('deactivating_rate')) > 0 TODO #521
