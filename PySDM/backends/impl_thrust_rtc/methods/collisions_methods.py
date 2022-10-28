"""
GPU implementation of backend methods for particle collisions
"""
from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods

COMMONS = """
struct Commons {
  static __device__ void flag_zero_multiplicity(
    int64_t j,
    int64_t k,
    VectorView<int64_t> multiplicity,
    VectorView<int64_t> healthy
  ) {
    if (multiplicity[k] == 0 || multiplicity[j] == 0) {
        healthy[0] = 0;
    }
  }
};
"""


class CollisionsMethods(
    ThrustRTCBackendMethods
):  # pylint: disable=too-many-instance-attributes
    def __init__(self):
        ThrustRTCBackendMethods.__init__(self)
        const = self.formulae.constants

        self.__adaptive_sdm_gamma_body_1 = trtc.For(
            ("dt_todo", "dt_left", "dt_max"),
            "cid",
            """
                dt_todo[cid] = min(dt_left[cid], dt_max);
            """,
        )

        self.__adaptive_sdm_gamma_body_2 = trtc.For(
            ("gamma", "idx", "n", "cell_id", "dt", "is_first_in_pair", "dt_todo"),
            "i",
            """
                if (gamma[i] == 0) {
                    return;
                }
                auto offset = 1 - is_first_in_pair[2 * i];
                auto j = idx[2 * i + offset];
                auto k = idx[2 * i + 1 + offset];
                auto prop = (int64_t)(n[j] / n[k]);
                auto dt_optimal = dt * prop / gamma[i];
                auto cid = cell_id[j];
                static_assert(sizeof(dt_todo[0]) == sizeof(unsigned int), "");
                atomicMin((unsigned int*)&dt_todo[cid], __float_as_uint(dt_optimal));
            """,
        )

        self.__adaptive_sdm_gamma_body_3 = trtc.For(
            ("gamma", "idx", "cell_id", "dt", "is_first_in_pair", "dt_todo"),
            "i",
            """
                if (gamma[i] == 0) {
                    return;
                }
                auto offset = 1 - is_first_in_pair[2 * i];
                auto j = idx[2 * i + offset];
                gamma[i] *= dt_todo[cell_id[j]] / dt;
            """,
        )

        self.__adaptive_sdm_gamma_body_4 = trtc.For(
            ("dt_left", "dt_todo", "stats_n_substep"),
            "cid",
            """
                dt_left[cid] -= dt_todo[cid];
                if (dt_todo[cid] > 0) {
                    stats_n_substep[cid] += 1;
                }
            """,
        )

        self.___sort_by_cell_id_and_update_cell_start_body = trtc.For(
            ("cell_id", "cell_start", "idx"),
            "i",
            """
            if (i == 0) {
                cell_start[cell_id[idx[0]]] = 0;
            }
            else {
                auto cell_id_curr = cell_id[idx[i]];
                auto cell_id_next = cell_id[idx[i + 1]];
                auto diff = (cell_id_next - cell_id_curr);
                for (auto j = 1; j < diff + 1; j += 1) {
                    cell_start[cell_id_curr + j] = idx[i + 1];
                }
            }
            """,
        )

        self.__collision_coalescence_body = trtc.For(
            param_names=(
                "n",
                "idx",
                "n_sd",
                "attributes",
                "n_attr",
                "gamma",
                "healthy",
            ),
            name_iter="i",
            body=f"""
            {COMMONS}
            if (gamma[i] == 0) {{
                return;
            }}
            auto j = idx[2 * i];
            auto k = idx[2 * i + 1];

            auto new_n = n[j] - gamma[i] * n[k];
            if (new_n > 0) {{
                n[j] = new_n;
                for (auto attr = 0; attr < n_attr * n_sd; attr+=n_sd) {{
                    attributes[attr + k] += gamma[i] * attributes[attr + j];
                }}
            }}
            else {{  // new_n == 0
                n[j] = (int64_t)(n[k] / 2);
                n[k] = n[k] - n[j];
                for (auto attr = 0; attr < n_attr * n_sd; attr+=n_sd) {{
                    attributes[attr + j] = gamma[i] * attributes[attr + j] + attributes[attr + k];
                    attributes[attr + k] = attributes[attr + j];
                }}
            }}

            Commons::flag_zero_multiplicity(j, k, n, healthy);
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

        self.__compute_gamma_body = trtc.For(
            (
                "gamma",
                "rand",
                "idx",
                "n",
                "cell_id",
                "collision_rate_deficit",
                "collision_rate",
            ),
            "i",
            """
            gamma[i] = ceil(gamma[i] - rand[i]);

            if (gamma[i] == 0) {
                return;
            }

            auto j = idx[2 * i];
            auto k = idx[2 * i + 1];

            auto prop = (int64_t)(n[j] / n[k]);
            if (prop > gamma[i]) {
                prop = gamma[i];
            }

            gamma[i] = prop;
            """,
        )

        self.__normalize_body_0 = trtc.For(
            ("cell_start", "norm_factor", "dt_div_dv"),
            "i",
            """
            auto sd_num = cell_start[i + 1] - cell_start[i];
            if (sd_num < 2) {
                norm_factor[i] = 0;
            }
            else {
                auto half_sd_num = sd_num / 2;
                norm_factor[i] = dt_div_dv * sd_num * (sd_num - 1) / 2 / half_sd_num;
            }
            """,
        )

        self.__normalize_body_1 = trtc.For(
            ("prob", "cell_id", "norm_factor"),
            "i",
            """
            prob[i] *= norm_factor[cell_id[i]];
            """,
        )

        self.__remove_zero_n_or_flagged_body = trtc.For(
            ("data", "idx", "n_sd"),
            "i",
            """
            if (idx[i] < n_sd && data[idx[i]] == 0) {
                idx[i] = n_sd;
            }
            """,
        )

        self.__cell_id_body = trtc.For(
            ("cell_id", "cell_origin", "strides", "n_dims", "size"),
            "i",
            """
            cell_id[i] = 0;
            for (auto j = 0; j < n_dims; j += 1) {
                cell_id[i] += cell_origin[size * j + i] * strides[j];
            }
            """,
        )

        self.__exp_fragmentation_body = trtc.For(
            param_names=("scale", "frag_size", "rand", "tol"),
            name_iter="i",
            body="""
            frag_size[i] = -scale * log(max(1 - rand[i], tol));
            """,
        )

        self.__fragmentation_limiters_body = trtc.For(
            param_names=(
                "n_fragment",
                "frag_size",
                "v_max",
                "x_plus_y",
                "vmin",
                "nfmax",
                "nfmax_is_not_none",
            ),
            name_iter="i",
            body="""
            frag_size[i] = min(frag_size[i], v_max[i]);
            frag_size[i] = max(frag_size[i], vmin);

            if (nfmax_is_not_none) {
                if (x_plus_y[i] / frag_size[i] > nfmax) {
                    frag_size[i] = x_plus_y[i] / nfmax;
                }
            }
            if (frag_size[i] == 0.0) {
                frag_size[i] = x_plus_y[i];
                n_fragment[i] = 1.0;
            }
            n_fragment[i] = x_plus_y[i] / frag_size[i];
            """,
        )

        self.__gauss_fragmentation_body = trtc.For(
            param_names=("mu", "sigma", "frag_size", "rand"),
            name_iter="i",
            body=f"""
            frag_size[i] = mu - sigma / {const.sqrt_two} / {const.sqrt_pi} / log(2.) * log(
                (0.5 + rand[i]) / (1.5 - rand[i])
            );
            """,
        )

        self.__slams_fragmentation_body = trtc.For(
            param_names=("n_fragment", "frag_size", "x_plus_y", "probs", "rand"),
            name_iter="i",
            body="""
            probs[i] = 0.0;
            n_fragment[i] = 1;

            for (auto n = 0; n < 22; n += 1) {
                probs[i] += 0.91 * pow((n + 2), (-1.56));
                if (rand[i] < probs[i]) {
                    n_fragment[i] = n + 2;
                    break;
                }
            }
            frag_size[i] = x_plus_y[i] / n_fragment[i];
            """,
        )

        self.__feingold1988_fragmentation_body = trtc.For(
            param_names=("scale", "frag_size", "x_plus_y", "rand", "fragtol"),
            name_iter="i",
            body="""
            auto log_arg = max(1 - rand[i] * scale / x_plus_y[i], fragtol);
            frag_size[i] = -scale * log(log_arg);
            """,
        )

        self.__straub_Nr_body = """

            if (gam[i] * CW[i] >= 7.0) {
                Nr1[i] = 0.088 * (gam[i] * CW[i] - 7.0);
            }
            if (CW[i] >= 21.0) {
                Nr2[i] = 0.22 * (CW[i] - 21.0);
                if (CW[i] <= 46.0) {
                    Nr3[i] = 0.04 * (46.0 - CW[i]);
                }
            }
            else {
                Nr3[i] = 1.0;
            }
            Nr4[i] = 1.0;
            Nrt[i] = Nr1[i] + Nr2[i] + Nr3[i] + Nr4[i];
        """

        self.__straub_p1 = f"""

                auto E_D1 = 0.04 * CM;
                auto delD1 = 0.0125 * pow(CW[i], (real_type) (0.5));
                auto var_1 = pow(delD1, 2.) / 12.;
                auto sigma1 = sqrt(log(var_1 / pow(E_D1, 2.) + 1));
                auto mu1 = log(E_D1) - pow(sigma1, 2.) / 2.;
                auto X = rand[i];

                frag_size[i] = exp(
                    mu1
                    - sigma1 / {const.sqrt_two} / {const.sqrt_pi} / log(2.) * log((0.5 + X) / (1.5 - X))
                );
                frag_size[i] = {const.PI} / 6. * pow(frag_size[i], (real_type) (3.));
        """.replace(
            "real_type", self._get_c_type()
        )

        self.__straub_p2 = f"""

                auto mu2 = 0.095 * CM;
                auto delD2 = 0.007 * (CW[i] - 21.0);
                auto sigma2 = pow(delD2, 2.) / 12.;
                auto X = rand[i];

                frag_size[i] = mu2 - sigma2 / {const.sqrt_two} / {const.sqrt_pi} / log(2.) * log(
                    (0.5 + X) / (1.5 - X)
                );
                frag_size[i] = {const.PI} / 6. * pow(frag_size[i], (real_type) (3.));
        """.replace(
            "real_type", self._get_c_type()
        )

        self.__straub_p3 = f"""

                auto mu3 = 0.9 * ds[i];
                auto delD3 = 0.01 * (0.76 * pow(CW[i], (real_type) (0.5)) + 1.0);
                auto sigma3 = pow(delD3, 2.) / 12.;
                auto X = rand[i];

                frag_size[i] = mu3 - sigma3 / {const.sqrt_two} / {const.sqrt_pi} / log(2.) * log(
                    (0.5 + X) / (1.5 - X)
                );
                frag_size[i] = {const.PI} / 6. * pow(frag_size[i], (real_type) (3.));
        """.replace(
            "real_type", self._get_c_type()
        )

        self.__straub_p4 = f"""

                auto E_D1 = 0.04 * CM;
                auto delD1 = 0.0125 * pow(CW[i], (real_type) (0.5));
                auto var_1 = pow(delD1, 2.) / 12.;
                auto sigma1 = sqrt(log(var_1 / pow(E_D1, 2.) + 1));
                auto mu1 = log(E_D1) - pow(sigma1, 2.) / 2.;
                auto mu2 = 0.095 * CM;
                auto delD2 = 0.007 * (CW[i] - 21.0);
                auto sigma2 = pow(delD2, 2.) / 12.;
                auto mu3 = 0.9 * ds[i];
                auto delD3 = 0.01 * (0.76 * pow(CW[i], (real_type) (0.5)) + 1.0);
                auto sigma3 = pow(delD3, 2.) / 12.;

                auto M31 = Nr1[i] * exp(3 * mu1 + 9 * pow(sigma1, 2.) / 2.);
                auto M32 = Nr2[i] * (pow(mu2, 3.) + 3 * mu2 * pow(sigma2, 2.));
                auto M33 = Nr3[i] * (pow(mu3, 3.) + 3 * mu3 * pow(sigma3, 2.));

                auto M34 = v_max[i] / {const.PI_4_3} * 8 + pow(ds[i], (real_type) (3.)) - M31 - M32 - M33;
                frag_size[i] = {const.PI} / 6. * M34;
        """.replace(
            "real_type", self._get_c_type()
        )

        self.__straub_fragmentation_body = trtc.For(
            param_names=(
                "CW",
                "gam",
                "ds",
                "frag_size",
                "v_max",
                "rand",
                "Nr1",
                "Nr2",
                "Nr3",
                "Nr4",
                "Nrt",
            ),
            name_iter="i",
            body=f"""
            auto CM = .01;
            {self.__straub_Nr_body}

            if (rand[i] < Nr1[i] / Nrt[i]) {{
                rand[i] = rand[i] * Nrt[i] / Nr1[i];
                {self.__straub_p1}
            }}
            else if (rand[i] < (Nr2[i] + Nr1[i]) / Nrt[i]) {{
                rand[i] = (rand[i] * Nrt[i] - Nr1[i]) / (Nr2[i] - Nr1[i]);
                {self.__straub_p2}
            }}
            else if (rand[i] < (Nr3[i] + Nr2[i] + Nr1[i]) / Nrt[i]) {{
                rand[i] = (rand[i] * Nrt[i] - Nr2[i]) / (Nr3[i] - Nr2[i]);
                {self.__straub_p3}
            }}
            else {{
                {self.__straub_p4}
            }}
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def adaptive_sdm_end(self, dt_left, cell_start):
        i = trtc.Find(dt_left.data, self._get_floating_point(0))
        if i is None:
            i = len(dt_left)
        return cell_start[i]

    # pylint: disable=unused-argument
    @nice_thrust(**NICE_THRUST_FLAGS)
    def adaptive_sdm_gamma(
        self,
        *,
        gamma,
        n,
        cell_id,
        dt_left,
        dt,
        dt_range,
        is_first_in_pair,
        stats_n_substep,
        stats_dt_min,
    ):
        # TODO #406 implement stats_dt_min
        dt_todo = trtc.device_vector("float", len(dt_left))
        d_dt_max = self._get_floating_point(dt_range[1])
        d_dt = self._get_floating_point(dt)

        self.__adaptive_sdm_gamma_body_1.launch_n(
            len(dt_left), (dt_todo, dt_left.data, d_dt_max)
        )
        self.__adaptive_sdm_gamma_body_2.launch_n(
            len(n) // 2,
            (
                gamma.data,
                n.idx.data,
                n.data,
                cell_id.data,
                d_dt,
                is_first_in_pair.indicator.data,
                dt_todo,
            ),
        )
        self.__adaptive_sdm_gamma_body_3.launch_n(
            len(n) // 2,
            (
                gamma.data,
                n.idx.data,
                cell_id.data,
                d_dt,
                is_first_in_pair.indicator.data,
                dt_todo,
            ),
        )
        self.__adaptive_sdm_gamma_body_4.launch_n(
            len(dt_left), (dt_left.data, dt_todo, stats_n_substep.data)
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def cell_id(self, cell_id, cell_origin, strides):
        if len(cell_id) == 0:
            return

        assert cell_origin.shape[0] == strides.shape[1]
        assert cell_id.shape[0] == cell_origin.shape[1]
        assert strides.shape[0] == 1
        n_dims = trtc.DVInt64(cell_origin.shape[0])
        size = trtc.DVInt64(cell_origin.shape[1])
        self.__cell_id_body.launch_n(
            len(cell_id), (cell_id.data, cell_origin.data, strides.data, n_dims, size)
        )

    # pylint: disable=unused-argument
    @nice_thrust(**NICE_THRUST_FLAGS)
    def collision_coalescence(
        self,
        *,
        multiplicity,
        idx,
        attributes,
        gamma,
        healthy,
        cell_id,
        coalescence_rate,
        is_first_in_pair,
    ):
        if len(idx) < 2:
            return
        n_sd = trtc.DVInt64(attributes.shape[1])
        n_attr = trtc.DVInt64(attributes.shape[0])
        self.__collision_coalescence_body.launch_n(
            len(idx) // 2,
            (
                multiplicity.data,
                idx.data,
                n_sd,
                attributes.data,
                n_attr,
                gamma.data,
                healthy.data,
            ),
        )

    # pylint: disable=unused-argument
    @nice_thrust(**NICE_THRUST_FLAGS)
    def compute_gamma(
        self,
        *,
        gamma,
        rand,
        multiplicity,
        cell_id,
        collision_rate_deficit,
        collision_rate,
        is_first_in_pair,
    ):
        if len(multiplicity) < 2:
            return
        self.__compute_gamma_body.launch_n(
            len(multiplicity) // 2,
            (
                gamma.data,
                rand.data,
                multiplicity.idx.data,
                multiplicity.data,
                cell_id.data,
                collision_rate_deficit.data,
                collision_rate.data,
            ),
        )

    # pylint: disable=unused-argument
    def make_cell_caretaker(self, idx, cell_start, scheme=None):
        return self._sort_by_cell_id_and_update_cell_start

    # pylint: disable=unused-argument
    @nice_thrust(**NICE_THRUST_FLAGS)
    def normalize(
        self, *, prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv
    ):
        n_cell = cell_start.shape[0] - 1
        device_dt_div_dv = self._get_floating_point(timestep / dv)
        self.__normalize_body_0.launch_n(
            n_cell, (cell_start.data, norm_factor.data, device_dt_div_dv)
        )
        self.__normalize_body_1.launch_n(
            prob.shape[0], (prob.data, cell_id.data, norm_factor.data)
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def remove_zero_n_or_flagged(self, data, idx, length) -> int:
        n_sd = trtc.DVInt64(idx.size())

        # Warning: (potential bug source): reading from outside of array
        self.__remove_zero_n_or_flagged_body.launch_n(length, [data, idx, n_sd])

        trtc.Sort(idx)

        result = idx.size() - trtc.Count(idx, n_sd)
        return result

    # pylint: disable=unused-argument
    @nice_thrust(**NICE_THRUST_FLAGS)
    def _sort_by_cell_id_and_update_cell_start(
        self, cell_id, cell_idx, cell_start, idx
    ):
        # TODO #330
        #   was here before (but did not work):
        #      trtc.Sort_By_Key(cell_id.data, idx.data)
        #   was here before (but cause huge slowdown of otherwise correct code)
        #      max_cell_id = max(cell_id.to_ndarray())
        #      assert max_cell_id == 0
        #   no handling of cell_idx in ___sort_by_cell_id_and_update_cell_start_body yet
        n_sd = cell_id.shape[0]
        trtc.Fill(cell_start.data, trtc.DVInt64(n_sd))
        if len(idx) > 1:
            self.___sort_by_cell_id_and_update_cell_start_body.launch_n(
                len(idx) - 1, (cell_id.data, cell_start.data, idx.data)
            )
        return idx

    def exp_fragmentation(
        self,
        *,
        n_fragment,
        scale,
        frag_size,
        v_max,
        x_plus_y,
        rand,
        vmin,
        nfmax,
        tol=1e-5,
    ):
        self.__exp_fragmentation_body.launch_n(
            n=len(frag_size),
            args=(
                self._get_floating_point(scale),
                frag_size.data,
                rand.data,
                self._get_floating_point(tol),
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_size),
            args=(
                n_fragment.data,
                frag_size.data,
                v_max.data,
                x_plus_y.data,
                self._get_floating_point(vmin),
                self._get_floating_point(nfmax if nfmax else -1),
                trtc.DVBool(nfmax is not None),
            ),
        )

    def gauss_fragmentation(
        self, *, n_fragment, mu, sigma, frag_size, v_max, x_plus_y, rand, vmin, nfmax
    ):
        self.__gauss_fragmentation_body.launch_n(
            n=len(frag_size),
            args=(
                self._get_floating_point(mu),
                self._get_floating_point(sigma),
                frag_size.data,
                rand.data,
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_size),
            args=(
                n_fragment.data,
                frag_size.data,
                v_max.data,
                x_plus_y.data,
                self._get_floating_point(vmin),
                self._get_floating_point(nfmax if nfmax else -1),
                trtc.DVBool(nfmax is not None),
            ),
        )

    def slams_fragmentation(
        self, n_fragment, frag_size, v_max, x_plus_y, probs, rand, vmin, nfmax
    ):  # pylint: disable=too-many-arguments
        self.__slams_fragmentation_body.launch_n(
            n=(len(n_fragment)),
            args=(
                n_fragment.data,
                frag_size.data,
                x_plus_y.data,
                probs.data,
                rand.data,
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_size),
            args=(
                n_fragment.data,
                frag_size.data,
                v_max.data,
                x_plus_y.data,
                self._get_floating_point(vmin),
                self._get_floating_point(nfmax if nfmax else -1),
                trtc.DVBool(nfmax is not None),
            ),
        )

    def feingold1988_fragmentation(
        self,
        *,
        n_fragment,
        scale,
        frag_size,
        v_max,
        x_plus_y,
        rand,
        fragtol,
        vmin,
        nfmax,
    ):
        self.__feingold1988_fragmentation_body.launch_n(
            n=len(frag_size),
            args=(
                self._get_floating_point(scale),
                frag_size.data,
                x_plus_y.data,
                rand.data,
                self._get_floating_point(fragtol),
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_size),
            args=(
                n_fragment.data,
                frag_size.data,
                v_max.data,
                x_plus_y.data,
                self._get_floating_point(vmin),
                self._get_floating_point(nfmax if nfmax else -1),
                trtc.DVBool(nfmax is not None),
            ),
        )

    def straub_fragmentation(
        # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        n_fragment,
        CW,
        gam,
        ds,
        frag_size,
        v_max,
        x_plus_y,
        rand,
        vmin,
        nfmax,
        Nr1,
        Nr2,
        Nr3,
        Nr4,
        Nrt,
    ):
        self.__straub_fragmentation_body.launch_n(
            n=len(frag_size),
            args=(
                CW.data,
                gam.data,
                ds.data,
                frag_size.data,
                v_max.data,
                rand.data,
                Nr1.data,
                Nr2.data,
                Nr3.data,
                Nr4.data,
                Nrt.data,
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_size),
            args=(
                n_fragment.data,
                frag_size.data,
                v_max.data,
                x_plus_y.data,
                self._get_floating_point(vmin),
                self._get_floating_point(nfmax if nfmax else -1),
                trtc.DVBool(nfmax is not None),
            ),
        )
