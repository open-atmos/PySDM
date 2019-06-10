"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.runner import Runner
from SDM.state import State
from SDM.colliders import SDM
from SDM.undertakers import Resize
from SDM.discretisations import logarithmic
from SDM.spectra import Exponential
from SDM.kernels import Golovin


class setup:
    m_min = 0
    m_max = 0
    n_sd = 0
    n_part = 0
    x_0 = 0
    dt = 0
    dv = 0
    b = 0


def TODO(s):
    spectrum = Exponential(s.n_part, s.x_0) # TODO: parameters
    state = State(*logarithmic(s.n_sd, spectrum, (s.m_min, s.m_max)))
    kernel = Golovin(setup.b)
    collider = SDM(kernel, s.dt, s.dv)
    undertaker = Resize()
    runner = Runner(state, (collider, undertaker))

    runner.run(0)
