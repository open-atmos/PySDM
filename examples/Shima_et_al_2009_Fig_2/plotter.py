"""
Created at 12.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from matplotlib import pyplot
from PySDM.utils import Physics
from PySDM.simulation.physics.constants import si


class Plotter:
    def __init__(self, setup, xrange):
        self.setup = setup

        self.x_bins = np.logspace(
            (np.log10(xrange[0])),
            (np.log10(xrange[1])),
            num=64,
            endpoint=True
        )
        self.r_bins = Physics.x2r(self.x_bins)

    def show(self):
        pyplot.show()

    def save(self, file):
        pyplot.savefig(file)

    def plot(self, state, t):
        s = self.setup

        if t == 0:
            analytic_solution = s.spectrum.size_distribution
        else:
            analytic_solution = lambda x: s.norm_factor * s.kernel.analytic_solution(
                x=x, t=t, x_0=s.X0, N_0=s.n_part
            )

        dm = np.diff(self.x_bins)
        dr = np.diff(self.r_bins)

        pdf_m_x = self.x_bins[:-1] + dm / 2
        pdf_m_y = analytic_solution(pdf_m_x)

        pdf_r_x = self.r_bins[:-1] + dr / 2
        pdf_r_y = pdf_m_y * dm / dr * pdf_r_x

        pyplot.plot(
            pdf_r_x * si.metres / si.micrometres,
            pdf_r_y * Physics.r2x(pdf_r_x) * s.rho / s.dv * si.kilograms / si.grams,
            color='black'
        )

        vals = np.empty(len(self.r_bins) - 1)
        tmp = np.empty((1,1))
        moment_0 = state.backend.array(1, dtype=int)
        moments = state.backend.array((1, 1), dtype=float)
        for i in range(len(vals)):
            state.moments(moment_0, moments, specs={'x': (1,)}, attr_range=(self.x_bins[i], self.x_bins[i + 1]))
            state.backend.download(moments, tmp)
            vals[i] = tmp[0, 0]
            state.backend.download(moment_0, tmp)
            vals[i] *= tmp[0, 0]
            vals[i] *= s.rho / s.dv
            vals[i] /= (np.log(self.r_bins[i + 1]) - np.log(self.r_bins[i]))

        pyplot.step(
            self.r_bins[:-1] * si.metres / si.micrometres,
            vals * si.kilograms / si.grams,
            where='post',
            label=f"t = {t}s"
        )
        pyplot.grid()
        pyplot.xscale('log')
        pyplot.xlabel('particle radius [µm]')
        pyplot.ylabel('dm/dlnr [g/m^3/(unit dr/r)]')
        pyplot.legend()
