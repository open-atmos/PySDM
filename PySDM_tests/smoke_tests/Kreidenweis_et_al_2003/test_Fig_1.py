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
            pyplot.plot(example_output[name], np.asarray(example_output['t']) - Z_CB * si.s)
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
        assert 0.03 < example_output['aq_S_IV_ppb'][-1] + example_output['gas_S_IV_ppb'][-1] < 0.05

    @staticmethod
    def test_c(example_output, plot=False):
        if plot:
            pyplot.plot(example_output['pH'], np.asarray(example_output['t']) - Z_CB * si.s)
            pyplot.xlabel('pH')
            pyplot.show()

        assert 4.9 < example_output['pH_pH_number_weighted'][-1] < 5
        assert 4.9 < example_output['pH_pH_volume_weighted'][-1] < 5
        assert 4.7 < example_output['pH_conc_H_number_weighted'][-1] < 4.9
        assert 4.7 < example_output['pH_conc_H_volume_weighted'][-1] < 4.9
