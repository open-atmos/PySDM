"""
Created at 25.09.2019
"""

from PySDM_examples.ICMW_2012_case_1.settings import Settings
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1.storage import Storage
from PySDM_examples.ICMW_2012_case_1.export import netCDF


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

    exporter = netCDF(storage, settings, simulation)
    exporter.run()


if __name__ == '__main__':
    main()
