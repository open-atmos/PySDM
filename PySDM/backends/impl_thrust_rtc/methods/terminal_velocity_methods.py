"""
GPU implementation of backend methods for terminal velocities
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class TerminalVelocityMethods(ThrustRTCBackendMethods):
    @cached_property
    def __linear_collection_efficiency_body(self):
        return trtc.For(
            (
                "A",
                "B",
                "D1",
                "D2",
                "E1",
                "E2",
                "F1",
                "F2",
                "G1",
                "G2",
                "G3",
                "Mf",
                "Mg",
                "output",
                "radii",
                "is_first_in_pair",
                "idx",
                "unit",
            ),
            "i",
            """
            if (is_first_in_pair[i]) {
                real_type r = 0;
                real_type r_s = 0;
                if (radii[idx[i]] > radii[idx[i + 1]]) {
                    r = radii[idx[i]] / unit;
                    r_s = radii[idx[i + 1]] / unit;
                }
                else {
                    r = radii[idx[i + 1]] / unit;
                    r_s = radii[idx[i]] / unit;
                }
                real_type p = r_s / r;
                if (p != 0 && p != 1) {
                    real_type G = pow((G1 / r), Mg) + G2 + G3 * r;
                    real_type Gp = pow((1 - p), G);
                    if (Gp != 0) {
                        real_type D = D1 / pow(r, D2);
                        real_type E = E1 / pow(r, E2);
                        real_type F = pow((F1 / r), Mf) + F2;
                        output[int(i / 2)] = A + B * p + D / pow(p, F) + E / Gp;
                        if (output[int(i / 2)] < 0) {
                            output[int(i / 2)] = 0;
                        }
                    }
                }
            }
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __interpolation_body(self):
        # TODO #599 r<0
        return trtc.For(
            ("output", "radius", "factor", "a", "b"),
            "i",
            """
            auto r_id = (int64_t)(factor * radius[i]);
            auto r_rest = (factor * radius[i] - r_id) / factor;
            output[i] = a[r_id] + r_rest * b[r_id];
            """,
        )

    def linear_collection_efficiency(
        self, *, params, output, radii, is_first_in_pair, unit
    ):
        # pylint: disable=too-many-locals
        A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = params
        dA = self._get_floating_point(A)
        dB = self._get_floating_point(B)
        dD1 = self._get_floating_point(D1)
        dD2 = self._get_floating_point(D2)
        dE1 = self._get_floating_point(E1)
        dE2 = self._get_floating_point(E2)
        dF1 = self._get_floating_point(F1)
        dF2 = self._get_floating_point(F2)
        dG1 = self._get_floating_point(G1)
        dG2 = self._get_floating_point(G2)
        dG3 = self._get_floating_point(G3)
        dMf = self._get_floating_point(Mf)
        dMg = self._get_floating_point(Mg)
        dunit = self._get_floating_point(unit)
        trtc.Fill(output.data, trtc.DVDouble(0))
        self.__linear_collection_efficiency_body.launch_n(
            len(is_first_in_pair) - 1,
            [
                dA,
                dB,
                dD1,
                dD2,
                dE1,
                dE2,
                dF1,
                dF2,
                dG1,
                dG2,
                dG3,
                dMf,
                dMg,
                output.data,
                radii.data,
                is_first_in_pair.indicator.data,
                radii.idx.data,
                dunit,
            ],
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def interpolation(self, *, output, radius, factor, b, c):
        factor_device = trtc.DVInt64(factor)
        self.__interpolation_body.launch_n(
            len(radius), (output.data, radius.data, factor_device, b.data, c.data)
        )

    @cached_property
    def __power_series_body(self):
        # TODO #599 r<0
        return trtc.For(
            ("output", "radius", "num_terms", "prefactors", "powers"),
            "i",
            """
            output[i] = 0.0
            int kmax = num_terms
            for (auto k = 0; k < kmax; k+=1) {
                auto sumterm = prefactors[k] * pow(radius[i], 3*powers[k])
                output[i] = output[i] + sumterm
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def power_series(self, *, values, radius, num_terms, prefactors, powers):
        prefactors = self._get_floating_point(prefactors)
        powers = self._get_floating_point(powers)
        num_terms = self._get_floating_point(num_terms)
        self.__power_series_body.launch_n(
            values.size(), (values, radius, num_terms, prefactors, powers)
        )

    @cached_property
    def __terminal_velocity_body(self):
        return trtc.For(
            param_names=("values", "radius"),
            name_iter="i",
            body=f"""
            values[i] = {self.formulae.terminal_velocity.v_term.c_inline(
                radius="radius[i]"
            )};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def terminal_velocity(self, *, values, radius):
        self.__terminal_velocity_body.launch_n(n=values.size(), args=[values, radius])
