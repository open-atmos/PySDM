import os

from matplotlib import pyplot as plt
from PySDM_examples.Shima_et_al_2009.settings import Settings

from PySDM.backends import Numba, ThrustRTC
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products import WallTime


def run(settings, backend):
    env = Box(dv=settings.dv, dt=settings.dt)
    builder = Builder(n_sd=settings.n_sd, backend=backend, environment=env)
    attributes = {}
    sampling = ConstantMultiplicity(settings.spectrum)
    attributes["volume"], attributes["multiplicity"] = sampling.sample(settings.n_sd)
    builder.add_dynamic(Coalescence(collision_kernel=settings.kernel))
    particles = builder.build(attributes, products=(WallTime(),))

    states = {}
    last_wall_time = None
    for step in settings.output_steps:
        particles.run(step - particles.n_steps)
        last_wall_time = particles.products["wall time"].get()

    return states, last_wall_time


def main(plot: bool):
    settings = Settings()
    settings.steps = [100, 3600] if "CI" not in os.environ else [1, 2]

    times = {}
    for backend in (ThrustRTC, Numba):
        nsds = [2**n for n in range(12, 19, 3)]
        key = backend.__name__
        times[key] = []
        for sd in nsds:
            settings.n_sd = sd
            _, wall_time = run(settings, backend())
            times[key].append(wall_time)

    for backend, t in times.items():
        plt.plot(nsds, t, label=backend, linestyle="--", marker="o")
    plt.ylabel("wall time [s]")
    plt.xlabel("number of particles")
    plt.grid()
    plt.legend()
    plt.loglog(base=2)
    if plot:
        plt.show()


if __name__ == "__main__":
    main(plot="CI" not in os.environ)
