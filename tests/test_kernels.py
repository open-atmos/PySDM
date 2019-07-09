"""
Created at 07.06.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from SDM.kernels import Golovin
import numpy as np


class TestGolovin:
    def test_analytic_solution(self):
        b = 1.5e3
        x_0 = 4/3 * np.pi * 30.531e-6**3
        N_0 = 2**23

        sut = Golovin(b)

        from matplotlib import pyplot
        x = np.linspace(8e-25, 500e-15, 100)
        print(sut.analytic_solution(x=1e-10, t=0.01, x_0=x_0, N_0=N_0))
        pyplot.plot(x, sut.analytic_solution(x=x, t=0.01, x_0=x_0, N_0=N_0))
        pyplot.show()
