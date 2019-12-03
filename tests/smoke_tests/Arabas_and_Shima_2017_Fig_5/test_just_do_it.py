from examples.Arabas_and_Shima_2017_Fig_5.example import Simulation, setups
import pytest


@pytest.mark.parametrize("setup_idx", range(len(setups)))
def test_just_do_it(setup_idx):
    # Arrange
    setup = setups[setup_idx]
    simulation = Simulation(setup)

    # Act
    simulation.run()

    # Assert
    # TODO