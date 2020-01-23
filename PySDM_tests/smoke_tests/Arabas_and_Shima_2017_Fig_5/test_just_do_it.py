from PySDM_examples.Arabas_and_Shima_2017_Fig_5.example import Simulation, setups
import pytest


@pytest.mark.parametrize("setup_idx", range(len(setups)))
@pytest.mark.parametrize("scheme", ['BDF', 'libcloud'])
def test_just_do_it(setup_idx, scheme):
    # Arrange
    setup = setups[setup_idx]
    setup.scheme = scheme
    simulation = Simulation(setup)

    # Act
    simulation.run()

    # Assert
    # TODO: e.g., relative posiion of curves