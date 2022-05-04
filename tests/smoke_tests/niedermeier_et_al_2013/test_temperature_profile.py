# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from matplotlib import pyplot
import pytest
from PySDM_examples.Niedermeier_et_al_2014 import Settings, Simulation
from PySDM import Formulae
from PySDM.physics import si
from PySDM.physics.heterogeneous_ice_nucleation_rate import abifm

abifm.m = 54.48
abifm.c = -10.67


@pytest.mark.parametrize('initial_temperature', (280 * si.K, 270 * si.K))
def test_temperature_profile(initial_temperature, plot=False):
    # arrange
    formulae = Formulae(heterogeneous_ice_nucleation_rate='ABIFM')
    settings = Settings(
        initial_temperature=initial_temperature,
        timestep=10 * si.s,
        formulae=formulae
    )
    simulation = Simulation(settings)

    # act
    output = simulation.run()

    # plot
    pyplot.plot(output['T'], output['z'])
    if plot:
        pyplot.show()

    # assert
    assert output['T'][0] == initial_temperature
    assert output['T'][-1] < initial_temperature
