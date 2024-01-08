# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
from matplotlib import pyplot
from PySDM_examples.Niedermeier_et_al_2014 import Settings, Simulation

from PySDM import Formulae
from PySDM.physics import si


@pytest.mark.parametrize("initial_temperature", (280 * si.K, 270 * si.K))
def test_temperature_profile(initial_temperature, plot=False):
    # arrange
    formulae = Formulae(
        particle_shape_and_density="MixedPhaseSpheres",
        heterogeneous_ice_nucleation_rate="ABIFM",
        constants={"ABIFM_M": 54.48, "ABIFM_C": -10.67},
    )
    settings = Settings(
        initial_temperature=initial_temperature, timestep=10 * si.s, formulae=formulae
    )
    simulation = Simulation(settings)

    # act
    output = simulation.run()

    # plot
    pyplot.plot(output["T"], output["z"])
    if plot:
        pyplot.show()

    # assert
    assert abs(output["T"][0] - initial_temperature) < 1e-10 * si.K
    assert output["T"][-1] < initial_temperature
