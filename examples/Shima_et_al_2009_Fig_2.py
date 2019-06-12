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
from SDM.discretisations import logarithmic
from SDM.spectra import Lognormal
from SDM.kernels import Golovin


class setup:
    m_min = 10e-6
    m_max = 5000e-6
    n_sd = 2 ** 13
    n_part = 2 ** 23  # [m-3]
    m_mode = 50e-6  # [m] TODO: lognormal -> exp.
    s_geom = 3  # TODO: lognormal -> exp.
    dt = 1  # [s]
    dv = 1e6  # [m3]
    b = 1.5e3  # [s-1]


def test():
    s = setup
    spectrum = Lognormal(n_part=s.n_part, m_mode=s.m_mode, s_geom=s.s_geom)  # TODO: parameters
    state = State(*logarithmic(s.n_sd, spectrum, (s.m_min, s.m_max)))  # TODO: constant_multiplicity
    kernel = Golovin(setup.b)
    collider = SDM(kernel, s.dt, s.dv)
    undertaker = Resize()
    runner = Runner(state, (undertaker, collider))
    # TODO: plot 0, 1200, 2400, 3600
    for i in range(4):
        plot(state)
        runner.run(5)
    pyplot.show()


def plot(state):
    state.sort_by_m()
    pyplot.semilogx(state.m, state.n)
    pyplot.grid()
