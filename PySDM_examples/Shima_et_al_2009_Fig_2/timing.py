"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.particlesbuilder import ParticlesBuilder
from PySDM.simulation.dynamics.coalescence.algorithms.sdm import SDM
from PySDM.simulation.environment.box import Box
from PySDM.simulation.initialisation.spectral_sampling import constant_multiplicity
from PySDM_examples.Shima_et_al_2009_Fig_2.setup import SetupA


def run(setup):
    particles_building = ParticlesBuilder(n_sd=setup.n_sd, dt=setup.dt, backend=setup.backend)
    particles_building.set_mesh_0d(setup.dv)
    particles_building.set_environment(Box, {})
    v, n = constant_multiplicity(setup.n_sd, setup.spectrum, (setup.x_min, setup.x_max))
    particles_building.create_state_0d(n=n, extensive={'volume': v}, intensive={})
    particles_building.register_dynamic(SDM, {"kernel": setup.kernel})
    particles = particles_building.get_particles()

    states = {}
    for step in setup.steps:
        particles.run(step - particles.n_steps)
        # setup.check(runner.state, runner.n_steps) TODO???

    return states, particles.stats

#%%
# TODO python -O
from PySDM.backends.numba.numba import Numba
from PySDM.backends.thrustRTC.thrustRTC import ThrustRTC

setup = SetupA()
setup.steps = [100, 3600]

times = {}
for backend in (Numba, ThrustRTC):
    setup.backend = backend
    nsds = [2 ** n for n in range(12, 20, 3)]
    key = backend.__name__
    times[key] = []
    for sd in nsds:
        setup.n_sd = sd
        _, stats = run(setup)
        times[key].append(stats.wall_times[-1])
#%%
from matplotlib import pyplot as plt
for backend, t in times.items():
    plt.plot(nsds, t, label=backend)
plt.legend()
plt.loglog()
plt.show()
