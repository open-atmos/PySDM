"""
Created at 22.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from examples.ICMW_2012_case_1.example import Simulation, Setup


class DummyStorage:
    def save(*_): pass
    def init(*_): pass


def test_single_timestep():
    # Arrange
    setup = Setup()
    setup.steps = [0, 1]
    simulation = Simulation(setup, DummyStorage())

    # Act
    simulation.run()

    # Assert


