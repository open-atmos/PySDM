"""
Created at 16.12.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM_examples.ICMW_2012_case_1.setup import Setup
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1.storage import Storage
import PySDM.conf


def main():
    setup = Setup()

    setup.grid = (25, 25)
    setup.steps = [100, 3600]
    setup.processes = {
        "advection": True,
        "coalescence": True,
        "condensation": False
    }
    # setup.croupier = 'local_FisherYates'; PySDM.conf.NUMBA_PARALLEL = True
    n_sd = range(20, 100, 20)

    times = {}
    for method in ('global_FisherYates', 'local_FisherYates'):
        times[method] = []
        setup.croupier = method
        for sd in n_sd:
            setup.n_sd_per_gridbox = sd
            storage = Storage()
            simulation = Simulation(setup, storage)
            stats = simulation.run()
            times[method].append(stats.wall_times[-1])

    from matplotlib import pyplot as plt
    for method, t in times.items():
        plt.plot(n_sd, t, label=method)
    plt.legend()
    plt.loglog()
    plt.show()


if __name__ == '__main__':
    main()
