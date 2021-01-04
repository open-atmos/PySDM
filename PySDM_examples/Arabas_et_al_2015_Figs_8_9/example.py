"""
Created at 25.09.2019
"""

from PySDM_examples.Arabas_et_al_2015_Figs_8_9.settings import Settings
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.simulation import Simulation
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.storage import Storage
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.netcdf_exporter import NetCDFExporter
from PySDM_examples.utils.temporary_file import TemporaryFile


def main():
    settings = Settings()

    settings.n_sd_per_gridbox = 25
    settings.grid = (25, 25)
    settings.n_steps = 1200
    settings.n_spin_up = settings.n_steps // 2

    storage = Storage()
    simulation = Simulation(settings, storage)
    simulation.reinit()
    simulation.run()

    temp_file = TemporaryFile('.nc')
    exporter = NetCDFExporter(storage, settings, simulation, temp_file.absolute_path)
    exporter.run()


if __name__ == '__main__':
    main()
