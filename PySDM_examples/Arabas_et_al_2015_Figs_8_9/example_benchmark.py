"""
Created at 16.12.2019
"""

from PySDM_examples.Arabas_et_al_2015_Figs_8_9.settings import Settings
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.simulation import Simulation
from PySDM_examples.Arabas_et_al_2015_Figs_8_9.storage import Storage
from PySDM.products.stats.timers import WallTime
import PySDM.backends.numba.conf
from PySDM.backends import CPU, GPU
import importlib


def reload_CPU_backend():
    importlib.reload(PySDM.backends.numba.impl._maths_methods)
    importlib.reload(PySDM.backends.numba.impl._algorithmic_methods)
    importlib.reload(PySDM.backends.numba.impl._storage_methods)
    importlib.reload(PySDM.backends.numba.impl._physics_methods)
    importlib.reload(PySDM.backends)
    from PySDM.backends import CPU


def main():
    settings = Settings()

    settings.grid = (25, 25)
    settings.n_steps = 100
    settings.outfreq = 10
    settings.processes = {
        "particle advection": True,
        "fluid advection": True,
        "coalescence": True,
        "condensation": False,
        "sedimentation": True,
    }
    settings.condensation_dt_max = .2

    n_sd = range(15, 16, 1)

    times = {}
    backends = [(CPU, "sync"), (CPU, "async")]
    if GPU.ENABLE:
        backends.append((GPU, "async"))
    for backend, mode in backends:
        if backend is CPU:
            PySDM.backends.numba.conf.NUMBA_PARALLEL = mode
            reload_CPU_backend()
        settings.backend = backend
        key = f"{backend} (mode={mode})"
        times[key] = []
        for sd in n_sd:
            settings.n_sd_per_gridbox = sd
            storage = Storage()
            simulation = Simulation(settings, storage)
            simulation.reinit(products=[WallTime()])
            simulation.run()
            times[key].append(storage.load('wall_time')[-1])

    from matplotlib import pyplot as plt
    for parallelization, t in times.items():
        plt.plot(n_sd, t, label=parallelization)
    plt.legend()
    plt.loglog()
    plt.savefig("benchmark.pdf", format="pdf")


if __name__ == '__main__':
    main()
