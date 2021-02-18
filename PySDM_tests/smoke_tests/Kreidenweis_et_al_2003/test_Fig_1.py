from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.physics import si
from matplotlib import pyplot
import numpy as np


class TestFig1:
    @staticmethod
    def test_a(plot=True):
        # Arrange
        settings = Settings(n_sd=1, dt=1*si.s)
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Plot
        if plot:
            name = 'ql'
            prod = simulation.core.products['ql']
            pyplot.plot(output[name], np.asarray(output['t']) - 196 * si.s)
            pyplot.xlabel(f"{prod.name} [{prod.unit}]")
            pyplot.ylabel(f"time above cloud base [s]")
            pyplot.grid()
            pyplot.show()

        # Assert
        assert (np.diff(output['ql']) >= 0).all()

    @staticmethod
    def test_b(plot=True):
        # Arrange
        settings = Settings(n_sd=100, dt=1*si.s)
        simulation = Simulation(settings)

        # Act
        output = simulation.run()

        # Plot
        if plot:
            pyplot.plot(output['SO2_tot_conc'], np.asarray(output['t']) - 196 * si.s)
            pyplot.show()

        # Assert
        # TODO
