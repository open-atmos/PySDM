"""
Created at 12.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from matplotlib import pyplot
from PySDM.simulation.physics.constants import si
from PySDM.simulation.physics import formulae as phys


class Plotter:
    def __init__(self, setup):
        self.setup = setup

    def show(self):
        pyplot.show()

    def save(self, file):
        pyplot.savefig(file)

    def plot(self, vals, t):
        setup = self.setup

        if t == 0:
            analytic_solution = setup.spectrum.size_distribution
        else:
            analytic_solution = lambda x: setup.norm_factor * setup.kernel.analytic_solution(
                x=x, t=t, x_0=setup.X0, N_0=setup.n_part
            )

        volume_bins_edges = phys.volume(setup.radius_bins_edges)
        dm = np.diff(volume_bins_edges)
        dr = np.diff(setup.radius_bins_edges)

        pdf_m_x = volume_bins_edges[:-1] + dm / 2
        pdf_m_y = analytic_solution(pdf_m_x)

        pdf_r_x = setup.radius_bins_edges[:-1] + dr / 2
        pdf_r_y = pdf_m_y * dm / dr * pdf_r_x

        pyplot.plot(
            pdf_r_x * si.metres / si.micrometres,
            pdf_r_y * phys.volume(radius=pdf_r_x) * setup.rho / setup.dv * si.kilograms / si.grams,
            color='black'
        )

        pyplot.step(
            setup.radius_bins_edges[:-1] * si.metres / si.micrometres,
            vals * si.kilograms / si.grams,
            where='post',
            label=f"t = {t}s"
        )
        pyplot.grid()
        pyplot.xscale('log')
        pyplot.xlabel('particle radius [µm]')
        pyplot.ylabel('dm/dlnr [g/m^3/(unit dr/r)]')
        pyplot.legend()
