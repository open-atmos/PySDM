"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from matplotlib import pyplot
import numpy as np

from SDM.runner import Runner
from SDM.state import State
from SDM.colliders import SDM
from SDM.undertakers import Resize
from SDM.discretisations import constant_multiplicity, logarithmic
from SDM.spectra import Exponential
from SDM.kernels import Golovin


def x2r(x):
    return (x * 3/4 / np.pi)**(1/3)


def r2x(r):
    return 4/3 * np.pi * r**3


class Setup:
    m_min = r2x(8e-6)   # not given in the paper
    m_max = r2x(500e-6) # not given in the paper

    n_sd = 2 ** 17
    n_part = 2 ** 23  # [m-3]
    X0 = 4/3 * np.pi * 30.531e-6**3
    dt = 1  # [s]
    dv = 1e6  # [m3]
    b = 1.5e3  # [s-1]
    rho = 1000 # [kg m-3]

    check_LWC = 1e-3  # kg m-3  #TODO
    check_ksi = n_part * dv / n_sd  # TODO


def test():
    s = Setup # TODO: instantiation in principle not needed + usage in plot
    spectrum = Exponential(norm_factor=s.n_part * s.dv, scale=s.X0)
    state = State(*constant_multiplicity(s.n_sd, spectrum, (s.m_min, s.m_max)))

    assert np.min(state.n) == np.max(state.n)
    np.testing.assert_approx_equal(state.n[0], s.check_ksi, 1)

    kernel = Golovin(s.b)
    collider = SDM(kernel, s.dt, s.dv)
    undertaker = Resize()
    runner = Runner(state, (undertaker, collider))
    # TODO: plot 0, 1200, 2400, 3600
    plot(state, spectrum)
    for i in range(1):
        runner.run(1200)
        plot(state)
        # <TEMP> TODO
        LWC = s.rho * np.dot(state.n, state.m) / s.dv
        # </TEMP>
        np.testing.assert_approx_equal(LWC, s.check_LWC, 3)
    pyplot.show()


def plot(state, spectrum = None):
    x_bins = np.logspace(
        (np.log10(min(state.m))),
        (np.log10(max(state.m))),
        num=64,
        endpoint=True,
    )
    r_bins = x2r(x_bins)

    if spectrum is not None:
        dm = np.diff(x_bins)
        dr = np.diff(r_bins)

        pdf_m_x = x_bins[:-1] + dm/2
        pdf_m_y = spectrum.size_distribution(pdf_m_x)

        pdf_r_y = pdf_m_y * dm / dr
        pdf_r_x = r_bins[:-1] + dr / 2

        pyplot.plot(pdf_r_x, pdf_r_y * r2x(pdf_r_x) * Setup.rho / Setup.dv)

    vals = np.empty(len(r_bins)-1)
    for i in range(len(vals)):
        vals[i] = state.moment(1, (x_bins[i], x_bins[i+1]))
        vals[i] *= Setup.rho / Setup.dv
        vals[i] /= (r_bins[i+1] - r_bins[i])

    pyplot.step(r_bins[:-1], vals, where='post', label="hist")

    pyplot.xscale('log')
    pyplot.xlabel('particle radius [m]')
    pyplot.ylabel('TODO')
    pyplot.grid()
    #pyplot.xlim((m2r(Setup.m_min), m2r(Setup.m_max)))
    pyplot.legend()
