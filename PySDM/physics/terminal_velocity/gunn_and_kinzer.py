import numpy as np
from scipy.interpolate import Rbf
from PySDM.physics import constants as const


class Interpolation:
    def __init__(self, core, small_r_limit=None):
        self.core = core

        '''
        Gunn & Kinzer 1949, Table 2
        '''
        ir = np.array([.078, .1, .2, .3, .4, .5, .6, .7, .8 , .9, 1.0, 1.2, 1.4, 1.6,
                       1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2,
                       4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8
                       ]) * 1e-3 / 2
        iu = np.array([18, 27, 72, 117, 162, 206, 247, 287, 327, 367, 403, 464, 517, 565,
                       609, 649, 690, 727, 757, 782, 806, 826, 844, 860, 872, 883, 892,
                       898, 903, 907, 909, 912, 914, 916, 917
                       ]) / 100

        rbf = Rbf(ir, iu)
        self.factor = 100000
        num = 6 * self.factor//1000 + 1
        space, step = np.linspace(0, .6 * const.si.cm, num, retstep=True)
        u = np.empty(num)
        u[:] = rbf(space)
        u[0] = 0
        approximation_small = TpDependent.make(only_small=True)
        small_r_limit = small_r_limit or 40 * const.si.um
        approximation_small(u[1:], space[1:], small_r_limit)
        self.a = core.bck.Storage.from_ndarray(u)
        b = np.append(np.diff(u), [u[-1] - u[-2]]) / step
        self.b = core.bck.Storage.from_ndarray(b)

    def __call__(self, output, radius):
        self.core.backend.interpolation(output, radius, self.factor, self.a, self.b)


class RogersYau:
    """
    Rogers & Yau, equations: (8.5), (8.6), (8.8)
    """
    def __init__(self, particles, small_k=None, medium_k=None, large_k=None, small_r_limit=None, medium_r_limit=None):
        si = const.si
        self.particles = particles
        self.small_k = small_k or 1.19e6 / si.cm / si.s
        self.medium_k = medium_k or 8e3 / si.s
        self.large_k = large_k or 2.01e3 * si.cm ** (1 / 2) / si.s
        self.small_r_limit = small_r_limit or 35 * si.um
        self.medium_r_limit = medium_r_limit or 600 * si.um

    def __call__(self, output, radius):
        self.particles.backend.terminal_velocity(output.data, radius.data, self.small_k, self.medium_k, self.large_k,
                                                 self.small_r_limit, self.medium_r_limit)


# TODO #348 implement in backend logic
class TpDependent:
    def __init__(self, particles, small_r_limit):
        si = const.si
        self.small_r_limit = small_r_limit or 40 * si.um
        self.approximation = TpDependent.make()

    def __call__(self, output, radius):
        return self.approximation(output, radius, self.small_r_limit)

    @staticmethod
    def make(only_small=False):
        # TODO #348 T, p dependence
        # TODO #348 move constants to physics.constants

        si = const.si
        cm = si.cm
        T = 293.15
        p = 1000 * si.hPa

        p0 = 1013.25 * si.hPa
        rho0 = 1.204 * si.kg * si.m ** (-3)
        n = 1.832e-5  # * (1 + 0.00266 * (T - 296)) #* si.kg * si.m**(-1) * si.s**(-1)  # TODO #348
        rho = 0.348 * p / T  # * si.kg * si.m ** (-3)
        l0 = 6.62e-6 * si.cm
        n0 = 1.818e-5 * si.kg * si.m ** (-1) * si.s ** (-1)
        l = l0 * (n / n0) * (p0 * rho0 / p * rho) ** (1 / 2)  # TODO #348
        es = (n0 / n) - 1
        ec = ((rho0 / rho) ** (1 / 2)) - 1

        c4 = np.array([10.5035, 1.08750, -0.133245, -0.00659969])

        import numba
        from numba import prange
        from PySDM.backends.numba import conf

        @numba.njit(**{**conf.JIT_FLAGS, "cache": False, "parallel": False})
        def f4(r):
            return (n0 / n) * (1 + 1.255 * l / r) / (1 + 1.255 * l0 / r)

        c8 = np.array([6.5639, -1.0391, -1.4001, -0.82736, -0.34277, -0.083072, -0.010583, -0.00054208])

        @numba.njit(**{**conf.JIT_FLAGS, "cache": False, "parallel": False})
        def f8(r):
            result = (1.058 * ec - 1.104 * es)
            result *= (6.21 + np.log(r)) / 5.01
            result += 1.104 * es
            result += 1
            return result

        @numba.njit(**{**conf.JIT_FLAGS, "cache": False})
        def terminal_velocity(values, radius, threshold):
            for i in prange(len(values)):
                r = radius[i] / cm
                sum_r = 0
                if radius[i] < threshold:
                    for j in range(4):
                        sum_r += c4[j] * (np.log(2 * r) ** j)
                    values[i] = f4(r) * np.exp(sum_r) * cm
                elif not only_small:
                    for j in range(8):
                        sum_r += c8[j] * (np.log(2 * r) ** j)
                    values[i] = f8(r) * np.exp(sum_r) * cm

        return terminal_velocity
