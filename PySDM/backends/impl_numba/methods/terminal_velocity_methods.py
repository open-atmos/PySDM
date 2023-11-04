"""
CPU implementation of backend methods for terminal velocities
"""
from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.storages.numba import conf


class TerminalVelocityMethods(BackendMethods):
    @cached_property
    def interpolation_body(self):
        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def interpolation_body(output, radius, factor, b, c):
            for i in numba.prange(len(radius)):  # pylint: disable=not-an-iterable
                if radius[i] < 0:
                    output[i] = 0
                else:
                    r_id = int(factor * radius[i])
                    r_rest = ((factor * radius[i]) % 1) / factor
                    output[i] = b[r_id] + r_rest * c[r_id]

        return interpolation_body

    def interpolation(self, *, output, radius, factor, b, c):
        return self.interpolation_body(output.data, radius.data, factor, b.data, c.data)

    @cached_property
    def terminal_velocity_body(self):
        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def terminal_velocity_body(*, values, radius, k1, k2, k3, r1, r2):
            for i in numba.prange(len(values)):  # pylint: disable=not-an-iterable
                if radius[i] < r1:
                    values[i] = k1 * radius[i] ** 2
                elif radius[i] < r2:
                    values[i] = k2 * radius[i]
                else:
                    values[i] = k3 * radius[i] ** (1 / 2)

        return terminal_velocity_body

    def terminal_velocity(self, *, values, radius, k1, k2, k3, r1, r2):
        self.terminal_velocity_body(
            values=values, radius=radius, k1=k1, k2=k2, k3=k3, r1=r1, r2=r2
        )
