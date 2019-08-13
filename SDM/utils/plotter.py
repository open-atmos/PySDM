"""
Created at 12.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from matplotlib import pyplot

from SDM.utils.physics import Physics
from SDM.simulation.maths import Maths

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
            Physics.m2um * pdf_r_x,
            Physics.kg2g * pdf_r_y * Physics.r2x(pdf_r_x) * s.rho / s.dv,
            color='black'
        )

        vals = np.empty(len(self.r_bins) - 1)
        for i in range(len(vals)):
            vals[i] = Maths.moment(state, 1, attr='x', attr_range=(self.x_bins[i], self.x_bins[i + 1]))
            vals[i] *= s.rho / s.dv
            vals[i] /= (np.log(self.r_bins[i + 1]) - np.log(self.r_bins[i]))

        pyplot.step(
            Physics.m2um * self.r_bins[:-1],
            Physics.kg2g * vals,
            where='post',
            label=f"t = {t}s"
        )
        pyplot.grid()
        pyplot.xscale('log')
        pyplot.xlabel('particle radius [µm]')
        pyplot.ylabel('dm/dlnr [g/m^3/(unit dr/r)]')
        pyplot.legend()
