"""
Created at 07.02.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM_examples.ICMW_2012_case_1.export import netCDF
from PySDM_examples.ICMW_2012_case_1.setup import Setup
from PySDM_examples.ICMW_2012_case_1.storage import Storage
from PySDM_examples.ICMW_2012_case_1.simulation import DummyController
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation


def test_export():
    # Arrange
    setup = Setup()
    setup.steps = [1]

    storage = Storage()
    simulator = Simulation(setup, storage)
    sut = netCDF(storage, setup, simulator)
    controller = DummyController()

    simulator.reinit()
    simulator.run(controller)

    # Act
    sut.run(controller=controller)

    # Assert
    pass