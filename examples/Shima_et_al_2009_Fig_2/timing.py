"""
Created at 08.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.runner import Runner
from PySDM.simulation.state.state_factory import StateFactory
from PySDM.simulation.dynamics.coalescence import SDM
from PySDM.simulation.discretisations.spectral import constant_multiplicity
from examples.Shima_et_al_2009_Fig_2.setup import SetupA

def run(setup):
    x, n = constant_multiplicity(setup.n_sd, setup.spectrum, (setup.x_min, setup.x_max))
    state = StateFactory.state_0d(n=n, extensive={'x': x}, intensive={}, backend=setup.backend)
    collider = SDM(setup.kernel, setup.dt, setup.dv, n_sd=setup.n_sd, backend=setup.backend, n_cell=1)
    runner = Runner(state, (collider,))

    states = {}
    for step in setup.steps:
        runner.run(step - runner.n_steps)
        # setup.check(runner.state, runner.n_steps) TODO???

    return states, runner.stats


# TODO python -O
from PySDM.backends.numba.numba import Numba
from PySDM.backends.thrustRTC import ThrustRTC

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
#%%