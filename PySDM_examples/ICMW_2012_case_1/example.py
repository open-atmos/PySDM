"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM_examples.ICMW_2012_case_1.setup import Setup
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1.storage import Storage
import numpy as np


def main():
    with np.errstate(all='ignore'):
        setup = Setup()

        # TODO
        setup.n_sd_per_gridbox = 10
        setup.grid = (50, 50)
        setup.processes["coalescence"] = False
        setup.processes["condensation"] = True
        setup.condensation_rtol_lnv = 1e-4
        setup.condensation_rtol_thd = 1e-5
        setup.mpdata_iters = 3

        storage = Storage()
        simulation = Simulation(setup, storage)
        simulation.run()


if __name__ == '__main__':
    main()
