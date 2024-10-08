"""
CPU implementation of backend methods for terminal velocities
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class TerminalVelocityMethods(BackendMethods):
    @cached_property
    def _interpolation_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(output, radius, factor, b, c):
            for i in numba.prange(len(radius)):  # pylint: disable=not-an-iterable
                if radius[i] < 0:
                    output[i] = 0
                else:
                    r_id = int(factor * radius[i])
                    r_rest = ((factor * radius[i]) % 1) / factor
                    output[i] = b[r_id] + r_rest * c[r_id]

        return body

    def interpolation(self, *, output, radius, factor, b, c):
        return self._interpolation_body(
            output.data, radius.data, factor, b.data, c.data
        )

    @cached_property
    def _terminal_velocity_body(self):
        v_term = self.formulae.terminal_velocity.v_term

        @numba.njit(**self.default_jit_flags)
        def body(*, values, radius):
            for i in numba.prange(len(values)):  # pylint: disable=not-an-iterable
                values[i] = v_term(radius[i])

        return body

    def terminal_velocity(self, *, values, radius):
        self._terminal_velocity_body(values=values, radius=radius)

    @cached_property
    def _power_series_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(*, values, radius, num_terms, prefactors, powers):
            for i in numba.prange(len(values)):  # pylint: disable=not-an-iterable
                values[i] = 0.0
                for j in range(num_terms):
                    values[i] = values[i] + prefactors[j] * radius[i] ** (powers[j] * 3)

        return body

    def power_series(self, *, values, radius, num_terms, prefactors, powers):
        self._power_series_body(
            values=values,
            radius=radius,
            num_terms=num_terms,
            prefactors=prefactors,
            powers=powers,
        )
