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


kg2g = 1e3
m2um = 1e6


def x2r(x):
    return (x * 3/4 / np.pi)**(1/3)


def r2x(r):
    return 4/3 * np.pi * r**3


class Setup:
    m_min = r2x(10e-6)   # not given in the paper
    m_max = r2x(100e-6) # not given in the paper

    n_sd = 2 ** 13
    n_part = 2 ** 23  # [m-3]
    X0 = 4/3 * np.pi * 30.531e-6**3
    dt = 1  # [s]
    dv = 1e6  # [m3]
    b = 1.5e3  # [s-1]
    rho = 1000  # [kg m-3]

    check_LWC = 1e-3  # kg m-3  #TODO
    check_ksi = n_part * dv / n_sd  # TODO


def test():
    s = Setup  # TODO: instantiation in principle not needed + usage in plot
    norm_factor = s.n_part * s.dv
    spectrum = Exponential(norm_factor=norm_factor, scale=s.X0)
    state = State(*constant_multiplicity(s.n_sd, spectrum, (s.m_min, s.m_max)))

    assert np.min(state.n) == np.max(state.n)
    np.testing.assert_approx_equal(state.n[0], s.check_ksi, 1)

    kernel = Golovin(s.b)
    collider = SDM(kernel, s.dt, s.dv)
    undertaker = Resize()
    runner = Runner(state, (undertaker, collider))
    # TODO: plot 0, 1200, 2400, 3600
    plot(state, 0, spectrum.size_distribution)
    step = 250
    for i in range(3):
        t = (i+1)*step*s.dt
        runner.run(step)
        plot(state, t, lambda x: norm_factor*kernel.analytic_solution(
            x=x, t=t, x_0=s.X0, N_0=s.n_part
        ))
        # <TEMP> TODO
        LWC = s.rho * np.dot(state.n, state.m) / s.dv
        # </TEMP>
        np.testing.assert_approx_equal(LWC, s.check_LWC, 3)
    pyplot.show()


def plot(state, t, analytic_solution=None):
    print(max(state.m))
    x_bins = np.logspace(
        (np.log10(min(state.m))),
        (np.log10(max(state.m))),
        num=32,
        endpoint=True,
    )
    r_bins = x2r(x_bins)

    if analytic_solution is not None:
        dm = np.diff(x_bins)
        dr = np.diff(r_bins)

        pdf_m_x = x_bins[:-1] + dm/2
        pdf_m_y = analytic_solution(pdf_m_x)

        pdf_r_x = r_bins[:-1] + dr / 2
        pdf_r_y = pdf_m_y * dm / dr * pdf_r_x

        pyplot.plot(
            m2um * pdf_r_x,
            kg2g * pdf_r_y * r2x(pdf_r_x) * Setup.rho / Setup.dv,
            color='black'
        )

    vals = np.empty(len(r_bins)-1)
    for i in range(len(vals)):
        vals[i] = state.moment(1, (x_bins[i], x_bins[i+1]))
        vals[i] *= Setup.rho / Setup.dv
        vals[i] /= (np.log(r_bins[i+1]) - np.log(r_bins[i]))

    pyplot.step(m2um * r_bins[:-1], kg2g * vals, where='post', label=f"t = {t}s")
    pyplot.grid()
    pyplot.xscale('log')
    pyplot.xlabel('particle radius [µm]')
    pyplot.ylabel('dm/dlnr [g/m^3/(unit dr/r)]')
    pyplot.legend()

