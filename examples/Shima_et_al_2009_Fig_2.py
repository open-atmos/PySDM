"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from matplotlib import pyplot
from SDM.runner import Runner
from SDM.state import State
from SDM.colliders import SDM
from SDM.undertakers import Resize
from SDM.discretisations import logarithmic, constant_multiplicity
from SDM.spectra import Lognormal
from SDM.kernels import Golovin


class setup:
    m_min = 10e-6
    m_max = 5000e-6
    n_sd = 2 ** 13
    n_part = 2 ** 23  # [m-3]
    m_mode = 75e-6  # [m] TODO: lognormal -> exp.
    s_geom = 2  # TODO: lognormal -> exp.
    dt = 1  # [s]
    dv = 1e6  # [m3]
    b = 1.5e3  # [s-1]


def test():
    s = setup
    spectrum = Lognormal(n_part=s.n_part, m_mode=s.m_mode, s_geom=s.s_geom)  # TODO: parameters
    state = State(*constant_multiplicity(s.n_sd, spectrum, (s.m_min, s.m_max)))  # TODO: constant_multiplicity

    print("m", state.m)
    print("n", state.n)

    kernel = Golovin(setup.b)
    collider = SDM(kernel, s.dt, s.dv)
    undertaker = Resize()
    runner = Runner(state, (undertaker, collider))
    # TODO: plot 0, 1200, 2400, 3600
    plot(state, spectrum)
    for i in range(0):
        runner.run(5)
        plot(state)
    pyplot.show()


def plot(state, spectrum = None):

    import numpy as np
    bins = np.logspace(
        (np.log10(min(state.m))),
        (np.log10(max(state.m))),
        num=10,
        endpoint=True
    )

    if spectrum is not None:
        pyplot.plot(bins, spectrum.size_distribution(bins))

    vals = np.empty(len(bins)-1)
    for i in range(len(vals)):
        vals[i] = state.moment(0, (bins[i], bins[i+1]))
        vals[i] /= (bins[i+1] - bins[i])

    pyplot.step(bins[:-1], vals, where='post', label="hist")

    #state.sort_by_m()
    #pyplot.loglog(state.m, state.n)

    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.grid()
    pyplot.legend()
