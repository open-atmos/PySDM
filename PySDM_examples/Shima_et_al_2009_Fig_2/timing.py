"""
Created at 08.08.2019
"""

from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.spectral_sampling import constant_multiplicity
from PySDM_examples.Shima_et_al_2009_Fig_2.setup import SetupA


def run(setup):
    builder = Builder(n_sd=setup.n_sd, backend=setup.backend)
    builder.set_environment(Box(dv=setup.dv, dt=setup.dt))
    v, n = constant_multiplicity(setup.n_sd, setup.spectrum, (setup.init_x_min, setup.init_x_max))
    attributes = {'n': n, 'volume': v}
    builder.add_dynamic(Coalescence(setup.kernel))

    particles = builder.get_particles(attributes)

    states = {}
    for step in setup.steps:
        particles.run(step - particles.n_steps)

    return states, particles.stats


# TIP: try with: python -O
from PySDM.backends.numba.numba import Numba
from PySDM.backends.thrustRTC.thrustRTC import ThrustRTC

setup = SetupA()
setup.steps = [100, 3600]

times = {}
for backend in (ThrustRTC, Numba):
    setup.backend = backend
    nsds = [2 ** n for n in range(12, 19, 3)]
    key = backend.__name__
    times[key] = []
    for sd in nsds:
        setup.n_sd = sd
        _, stats = run(setup)
        times[key].append(stats.wall_times[-1])

from matplotlib import pyplot as plt
for backend, t in times.items():
    plt.plot(nsds, t, label=backend)
plt.legend()
plt.loglog()
plt.show()
