"""
Created at 25.09.2019
"""

from PySDM_examples.ICMW_2012_case_1.setup import Setup
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1.storage import Storage
import numpy as np


def main():
    with np.errstate(all='ignore'):
        setup = Setup()

        setup.n_sd_per_gridbox = 25
        setup.grid = (25, 25)
        for key in setup.processes.keys():
            setup.processes[key] = True
        setup.processes["relaxation"] = False
        setup.condensation_rtol_lnv = 1e-8
        setup.condensation_rtol_thd = 1e-8
        setup.mpdata_iters = 2

        storage = Storage()
        simulation = Simulation(setup, storage)
        simulation.reinit()
        simulation.run()


if __name__ == '__main__':
    main()
