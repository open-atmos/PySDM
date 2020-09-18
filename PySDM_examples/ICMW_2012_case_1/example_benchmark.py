"""
Created at 16.12.2019
"""

from types import ModuleType
from PySDM_examples.ICMW_2012_case_1.setup import Setup
from PySDM_examples.ICMW_2012_case_1.simulation import Simulation
from PySDM_examples.ICMW_2012_case_1.storage import Storage
import PySDM.backends.numba.conf
import importlib


def reload_CPU_backend():
    importlib.reload(PySDM.backends.numba.impl._maths_methods)
    importlib.reload(PySDM.backends.numba.impl._algorithmic_methods)
    importlib.reload(PySDM.backends.numba.impl._storage_methods)
    importlib.reload(PySDM.backends.numba.impl._physics_methods)
    importlib.reload(PySDM.backends)
    from PySDM.backends import CPU


def main():
    setup = Setup()

    setup.grid = (25, 25)
    setup.n_steps = 100
    setup.outfreq = 10
    setup.processes = {
        "particle advection": True,
        "fluid advection": True,
        "coalescence": True,
        "condensation": True,
        "sedimentation": False,
    }
    setup.condensation_dt_max = .2

    n_sd = range(15, 17, 1)

    times = {}
    for parallel in (True,):
        PySDM.backends.numba.conf.NUMBA_PARALLEL = parallel
        reload_CPU_backend()
        for method in ('local',):
            key = f"{method} (parallel={parallel})"
            times[key] = []
            for sd in n_sd:
                setup.n_sd_per_gridbox = sd
                storage = Storage()
                simulation = Simulation(setup, storage)
                simulation.reinit()
                # simulation.core.croupier = method
                stats = simulation.run()
                times[key].append(stats.wall_times[-1])

    from matplotlib import pyplot as plt
    for method, t in times.items():
        plt.plot(n_sd, t, label=method)
    plt.legend()
    plt.loglog()
    plt.savefig("benchamrk.pdf", format="pdf")


if __name__ == '__main__':
    main()
