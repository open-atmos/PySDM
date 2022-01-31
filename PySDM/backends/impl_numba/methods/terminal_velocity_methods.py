"""
CPU implementation of backend methods for terminal velocities
"""
import numba
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_common.backend_methods import BackendMethods


class TerminalVelocityMethods(BackendMethods):
    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments,too-many-locals
    def linear_collection_efficiency_body(
            params, output, radii, is_first_in_pair, idx, length, unit
    ):
        A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = params
        output[:] = 0
        for i in numba.prange(length - 1):  # pylint: disable=not-an-iterable
            if is_first_in_pair[i]:
                if radii[idx[i]] > radii[idx[i + 1]]:
                    r = radii[idx[i]] / unit
                    r_s = radii[idx[i + 1]] / unit
                else:
                    r = radii[idx[i + 1]] / unit
                    r_s = radii[idx[i]] / unit
                p = r_s / r
                if p not in (0, 1):
                    G = (G1 / r) ** Mg + G2 + G3 * r
                    Gp = (1 - p) ** G
                    if Gp != 0:
                        D = D1 / r ** D2
                        E = E1 / r ** E2
                        F = (F1 / r) ** Mf + F2
                        output[i // 2] = A + B * p + D / p ** F + E / Gp
                        output[i // 2] = max(0, output[i // 2])

    def linear_collection_efficiency(self, params, output, radii, is_first_in_pair, unit):
        return self.linear_collection_efficiency_body(
            params, output.data, radii.data, is_first_in_pair.indicator.data,
            radii.idx.data, len(is_first_in_pair), unit)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def interpolation_body(output, radius, factor, b, c):
        for i in numba.prange(len(radius)):  # pylint: disable=not-an-iterable
            if radius[i] < 0:
                output[i] = 0
            else:
                r_id = int(factor * radius[i])
                r_rest = ((factor * radius[i]) % 1) / factor
                output[i] = b[r_id] + r_rest * c[r_id]

    def interpolation(self, output, radius, factor, b, c):
        return self.interpolation_body(
            output.data, radius.data, factor, b.data, c.data
        )
