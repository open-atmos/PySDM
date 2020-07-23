"""
Created at 07.02.2020
"""

from PySDM_examples.ICMW_2012_case_1.export import netCDF
from PySDM_examples.ICMW_2012_case_1.setup import Setup
from PySDM_examples.ICMW_2012_case_1.simulation import DummyController
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1.storage import Storage


def test_export():
    # Arrange
    setup = Setup()
    setup.n_steps = 1
    setup.outfreq = 1

    storage = Storage()
    simulator = Simulation(setup, storage)
    sut = netCDF(storage, setup, simulator)
    controller = DummyController()

    simulator.reinit()
    simulator.run(controller)

    # Act
    sut.run(controller=controller)

    # Assert
