"""
Created at 07.02.2020
"""

from PySDM_examples.ICMW_2012_case_1.export import netCDF
from PySDM_examples.ICMW_2012_case_1.settings import Settings
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1.storage import Storage


def test_export():
    # Arrange
    settings = Settings()
    settings.n_steps = 1
    settings.outfreq = 1

    storage = Storage()
    simulator = Simulation(settings, storage)
    sut = netCDF(storage, settings, simulator)

    simulator.reinit()
    simulator.run()

    # Act
    sut.run()

    # Assert
