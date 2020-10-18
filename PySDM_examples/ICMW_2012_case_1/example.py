"""
Created at 25.09.2019
"""

from PySDM_examples.ICMW_2012_case_1.setup import Setup
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1.storage import Storage
from PySDM_examples.ICMW_2012_case_1.export import netCDF


def main():
    setup = Setup()

    setup.n_sd_per_gridbox = 25
    setup.grid = (25, 25)
    setup.n_steps = 1200
    setup.n_spin_up = setup.n_steps // 2

    storage = Storage()
    simulation = Simulation(setup, storage)
    simulation.reinit()
    simulation.run()

    exporter = netCDF(storage, setup, simulation)
    exporter.run()


if __name__ == '__main__':
    main()
