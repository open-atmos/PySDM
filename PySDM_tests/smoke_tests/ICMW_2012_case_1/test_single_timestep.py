"""
Created at 22.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1.setup import Setup
import numpy as np
from matplotlib import pyplot


class DummyStorage:
    def __init__(self):
        self.profiles = {}

    def init(*_): pass

    def save(self, data: np.ndarray, step: int, name: str):
        if name == "qv":
            self.profiles[step] = {"qv": np.mean(data, axis=0)}


def test_single_timestep():
    # Arrange
    setup = Setup()
    setup.steps = [0, 1]
    simulation = Simulation(setup, DummyStorage())

    # Act
    simulation.run()

    # Assert


def test_multi_timestep(plot=False):
    # Arrange
    setup = Setup()
    setup.steps = range(0, 20, 1)
    storage = DummyStorage()
    simulation = Simulation(setup, storage)

    # Act
    simulation.run()

    # Plot
    if plot:
        levels = np.arange(setup.grid[1])
        for step, datum in storage.profiles.items():
            pyplot.plot(datum["qv"], levels, label=str(step))
        pyplot.legend()
        pyplot.show()

    # Assert
    for step in range(len(storage.profiles)-1):
        next = storage.profiles[step+1]["qv"]
        prev = storage.profiles[step]["qv"]
        eps = 1e-5
        assert ((prev + eps) >= next).all()





