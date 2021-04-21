from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.physics.aqueous_chemistry.support import GASEOUS_COMPOUNDS
from PySDM.physics import si
from matplotlib import pyplot
import numpy as np
import pytest


@pytest.fixture(scope='session')
def example_output():
    settings = Settings(n_sd=16, dt=1*si.s, n_substep=5)
    simulation = Simulation(settings)
    output = simulation.run()
    return output


Z_CB = 196 * si.m


class TestFig1:
    @staticmethod
    def test_a(example_output, plot=False):
        # Plot
        if plot:
            name = 'ql'
            #prod = simulation.core.products['ql']
            pyplot.plot(example_output[name], np.asarray(example_output['t']) - Z_CB * si.s)
            #pyplot.xlabel(f"{prod.name} [{prod.unit}]")  # TODO #442
            pyplot.ylabel(f"time above cloud base [s]")
            pyplot.grid()
            pyplot.show()

        # Assert
        assert (np.diff(example_output['ql']) >= 0).all()

    @staticmethod
    def test_b(example_output, plot=False):
        # Plot
        if plot:
            for key in GASEOUS_COMPOUNDS.keys():
                pyplot.plot(
                    np.asarray(example_output[f'aq_{key}_ppb']),
                    np.asarray(example_output['t']) - Z_CB * si.s, label='aq')
                pyplot.plot(
                    np.asarray(example_output[f'gas_{key}_ppb']),
                    np.asarray(example_output['t']) - Z_CB * si.s, label='gas')
                pyplot.plot(
                    np.asarray(example_output[f'aq_{key}_ppb']) + np.asarray(example_output[f'gas_{key}_ppb']),
                    np.asarray(example_output['t']) - Z_CB * si.s, label='sum')
                pyplot.legend()
                pyplot.xlabel(key + ' [ppb]')
                pyplot.xscale('log')
                pyplot.show()

            pyplot.plot(
                np.asarray(example_output[f'aq_S_VI_ppb']),
                np.asarray(example_output['t']) - Z_CB * si.s, label='S_VI')
            pyplot.xlabel('S_VI (aq) [ppb]')
            pyplot.show()

        # Assert
        # assert False  TODO #442

    @staticmethod
    def test_c(example_output, plot=False):
        if plot:
            pyplot.plot(example_output['pH'], np.asarray(example_output['t']) - Z_CB * si.s)
            pyplot.xlabel('pH')
            pyplot.show()

        assert 5 < example_output['pH'][-1] < 5.2
