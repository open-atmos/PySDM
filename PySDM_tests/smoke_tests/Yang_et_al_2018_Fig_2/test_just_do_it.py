from PySDM_examples.Yang_et_al_2018_Fig_2.example import Simulation
from PySDM_examples.Yang_et_al_2018_Fig_2.setup import Setup


# TODO: run for different atol, rtol, scheme, dt_max
# @pytest.mark.parametrize("scheme", ['BDF', 'libcloud'])
def test_just_do_it():
    # Arrange
    setup = Setup()
    # setup.condensation_scheme = scheme
    # setup.dt_max = 0.07
    simulation = Simulation(setup)

    # Act
    output = simulation.run()

    # Assert
    # TODO!
