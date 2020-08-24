"""
Created at 24.08.2020 by edejong
"""

from ._gravitational import Gravitational
from PySDM.physics import constants as const

class Constant:
    def __init__(self, kernel_const):
        self.kernel_const = kernel_const
        
    def __call__(self, output, is_first_in_pair):
        # TODO: stop stupidly summing over all particles
        output.sum_pair(self.core.state['volume'],is_first_in_pair)
        output *= 0
        output += self.kernel_const
        print(self.kernel_const)
        
    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('volume')
        
# TODO: linear kernel

"""
class Geometric(Gravitational):

    def __init__(self, collection_efficiency=1, x='volume'):
        super().__init__()
        self.collection_efficiency = collection_efficiency
        self.x = x
        self.collection_efficiency = 1

    def __call__(self, output, is_first_in_pair):
        output.sum_pair(self.core.state['radius'], is_first_in_pair)
        output **= 2
        output *= const.pi * self.collection_efficiency
        self.tmp.distance_pair(self.core.state['terminal velocity'], is_first_in_pair)
        output *= self.tmp
        
        
class Golovin:

    def __init__(self, b):
        self.b = b
        self.core = None

    def __call__(self, output, is_first_in_pair):
        output.sum_pair(self.core.state['volume'], is_first_in_pair)
        output *= self.b

    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('volume')

    def analytic_solution(self, x, t, x_0, N_0):
        tau = 1 - mpmath.exp(-N_0 * self.b * x_0 * t)

        if isinstance(x, np.ndarray):
            func = np.vectorize(lambda i: Golovin.analytic_solution_helper(x[int(i)], tau, x_0))
            result = np.fromfunction(func, x.shape, dtype=float)
            return result

        return Golovin.analytic_solution_helper(x, tau, x_0)

    @staticmethod
    def analytic_solution_helper(x, tau, x_0):
        result = float(
            (1 - tau) *
            1 / (x * mpmath.sqrt(tau)) *
            mpmath.besseli(1, 2 * x / x_0 * mpmath.sqrt(tau)) *
            mpmath.exp(-(1 + tau) * x / x_0)
        )
        return result
"""