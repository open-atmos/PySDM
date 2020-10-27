"""
Created at 07.02.2020
"""

from PySDM_examples.Arabas_et_al_2015_Figs_8_9.netcdf_exporter import NetCDFExporter
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.settings import Settings
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.simulation import Simulation
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.storage import Storage


def test_export():
    # Arrange
    settings = Settings()
    settings.n_steps = 1
    settings.outfreq = 1

    storage = Storage()
    simulator = Simulation(settings, storage)
    sut = NetCDFExporter(storage, settings, simulator)

    simulator.reinit()
    simulator.run()

    # Act
    sut.run()

    # Assert
