"""
GPU implementation of backend methods for particle collisions
"""
import numpy as np

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust
from PySDM.physics.constants import sqrt_pi, sqrt_two

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class CollisionsMethods(
    ThrustRTCBackendMethods
):  # pylint: disable=too-many-instance-attributes
    def __init__(self):
        super().__init__()

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

        self.__coalescence_body = trtc.For(
            ("n", "idx", "n_sd", "attributes", "n_attr", "gamma", "healthy"),
            "i",
            """
            if (gamma[i] == 0) {
                return;
            }
            auto j = idx[2 * i];
            auto k = idx[2 * i + 1];

            auto new_n = n[j] - gamma[i] * n[k];
            if (new_n > 0) {
                n[j] = new_n;
                for (auto attr = 0; attr < n_attr * n_sd; attr+=n_sd) {
                    attributes[attr + k] += gamma[i] * attributes[attr + j];
                }
            }
            else {  // new_n == 0
                n[j] = (int64_t)(n[k] / 2);
                n[k] = n[k] - n[j];
                for (auto attr = 0; attr < n_attr * n_sd; attr+=n_sd) {
                    attributes[attr + j] = gamma[i] * attributes[attr + j] + attributes[attr + k];
                    attributes[attr + k] = attributes[attr + j];
                }
            }
            if (n[k] == 0 || n[j] == 0) {
                healthy[0] = 0;
            }
            """,
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
            param_names=("scale", "frag_size", "rand"),
            name_iter="i",
            body="""
            frag_size[i] = -scale * log(1 - rand[i]);
            """,
        )

        self.__fragmentation_limiters_body = trtc.For(
            param_names=(
                "n_fragment",
                "frag_size",
                "v_max",
                "x_plus_y",
                "vmin",
                "nfmax_is_not_none",
                "nfmax",
            ),
            name_iter="i",
            body="""
            n_fragment[i] = x_plus_y[i] / frag_size[i];
            if (frag_size[i] > v_max[i]) {
                n_fragment[i] = 1;
            }
            else if (frag_size[i] < vmin) {
                n_fragment[i] = 1;
            }

            if (nfmax_is_not_none) {
                n_fragment[i] = min(n_fragment[i], nfmax) / 0;
            }
            """,
        )

        self.__gauss_fragmentation_body = trtc.For(
            param_names=("mu", "sigma", "frag_size", "rand", "sqrt_pi", "sqrt_two"),
            name_iter="i",
            body="""
            frag_size[i] = mu + sqrt_pi * sqrt_two * sigma / 4 * log((1 + rand[i]) / (1 - rand[i]));
            """,
        )

        self.__slams_fragmentation_body = trtc.For(
            param_names=("n_fragment", "probs", "rand"),
            name_iter="i",
            body="""
            probs[i] = 0.0;
            n_fragment[i] = 1;

            for (auto n = 0; j < 22; j += 1) {
                probs[i] += 0.91 * pow((n + 2), (-1.56));
                if (rand[i] < probs[i]) {
                    n_fragment[i] = n + 2;
                    break;
                }
            }
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
        self.__coalescence_body.launch_n(
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
        self, *, n_fragment, scale, frag_size, v_max, x_plus_y, rand, vmin, nfmax
    ):
        self.__exp_fragmentation_body.launch_n(
            size=len(frag_size),
            args=(trtc.DVDouble(scale), frag_size.data, rand.data),
        )

        self.__fragmentation_limiters_body.launch_n(
            size=len(n_fragment),
            args=(
                n_fragment.data,
                frag_size.data,
                v_max.data,
                x_plus_y.data,
                trtc.DVDouble(vmin),
                trtc.DVBool(nfmax is not None),
                trtc.DVDouble(nfmax if nfmax else -1),
            ),
        )

    def gauss_fragmentation(
        self, *, n_fragment, mu, sigma, frag_size, v_max, x_plus_y, rand, vmin, nfmax
    ):
        print(sqrt_pi, sqrt_two)
        self.__gauss_fragmentation_body.launch_n(
            size=len(n_fragment),
            args=(
                trtc.DVDouble(mu),
                trtc.DVDouble(sigma),
                frag_size.data,
                rand.data,
                trtc.DVDouble(sqrt_pi),
                trtc.DVDouble(sqrt_two),
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            size=len(n_fragment),
            args=(
                n_fragment.data,
                frag_size.data,
                v_max.data,
                x_plus_y.data,
                trtc.DVDouble(vmin),
                trtc.DVBool(nfmax is not None),
                trtc.DVDouble(nfmax if nfmax else -1),
            ),
        )

    def slams_fragmentation(self, n_fragment, probs, rand):
        self.__slams_fragmentation_body.launch_n(
            size=(len(n_fragment)), args=(n_fragment.data, probs.data, rand.data)
        )
