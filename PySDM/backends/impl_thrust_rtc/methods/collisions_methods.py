"""
GPU implementation of backend methods for particle collisions
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods

# pylint: disable=too-many-lines

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

  static __device__ void coalesce(
    int64_t i,
    int64_t j,
    int64_t k,
    VectorView<int64_t> cell_id,
    VectorView<int64_t> multiplicity,
    VectorView<real_type> gamma,
    VectorView<real_type> attributes,
    VectorView<int64_t> coalescence_rate,
    int64_t n_attr,
    int64_t n_sd
  ) {
    auto cid = cell_id[j];
    atomicAdd((unsigned long long int*)&coalescence_rate[cid], (unsigned long long int)(gamma[i] * multiplicity[k]));
    auto new_n = multiplicity[j] - gamma[i] * multiplicity[k];
    if (new_n > 0) {
        multiplicity[j] = new_n;
        for (auto attr = 0; attr < n_attr * n_sd; attr+=n_sd) {
            attributes[attr + k] += gamma[i] * attributes[attr + j];
        }
    }
    else {  // new_n == 0
        multiplicity[j] = (int64_t)(multiplicity[k] / 2);
        multiplicity[k] = multiplicity[k] - multiplicity[j];
        for (auto attr = 0; attr < n_attr * n_sd; attr+=n_sd) {
            attributes[attr + j] = gamma[i] * attributes[attr + j] + attributes[attr + k];
            attributes[attr + k] = attributes[attr + j];
        }
    }
  }

  static __device__ auto pair_indices(
      int64_t i,
      int64_t *j,
      int64_t *k,
      VectorView<int64_t> idx,
      VectorView<bool> is_first_in_pair,
      VectorView<real_type> prob_like
  ) {
      if (prob_like[i] == 0) {
          return true;
      }
      auto offset = 1 - is_first_in_pair[2 * i];
      j[0] = idx[2 * i + offset];
      k[0] = idx[2 * i + 1 + offset];
      return false;
  }

  static __device__ auto compute_transfer_multiplicities(
    real_type gamma,
    int64_t j,
    int64_t k,
    VectorView<int64_t> multiplicity,
    VectorView<real_type> particle_mass,
    real_type fragment_mass_i,
    int64_t max_multiplicity,

    real_type *take_from_j,
    real_type *new_mult_k
  ) {
    real_type gamma_j_k = 0;
    real_type take_from_j_test = multiplicity[k];
    take_from_j[0] = 0;
    real_type new_mult_k_test = ((particle_mass[j] + particle_mass[k]) / fragment_mass_i * multiplicity[k]);
    new_mult_k[0] = multiplicity[k];

    for (auto m = 0; m < (int64_t) (gamma); m += 1) {
        if (new_mult_k_test > max_multiplicity) {
            break;
        }

        if (take_from_j_test > multiplicity[j]) {
            break;
        }

        take_from_j[0] = take_from_j_test;
        new_mult_k[0] = new_mult_k_test;
        gamma_j_k = m + 1;

        take_from_j_test += new_mult_k_test;
        new_mult_k_test = new_mult_k_test * (particle_mass[j] / fragment_mass_i) + new_mult_k_test;
    }
    return gamma_j_k;
  }

  static __device__ void get_new_multiplicities_and_update_attributes(
    int64_t j,
    int64_t k,
    VectorView<real_type> attributes,
    VectorView<int64_t> multiplicity,
    real_type take_from_j,
    real_type new_mult_k,
    int64_t n_attr,

    real_type *nj,
    real_type *nk
  ) {
    for (auto a = 0; a < n_attr; a +=1) {
        attributes[a + k] *= multiplicity[k];
        attributes[a + k] += take_from_j * attributes[a + j];
        attributes[a + k] /= new_mult_k;
    }

    if (multiplicity[j] > take_from_j) {
        nj[0] = multiplicity[j] - take_from_j;
        nk[0] = new_mult_k;
    } else {
        nj[0] = new_mult_k / 2;
        nk[0] = nj[0];

        for (auto a = 0; a < n_attr; a += 1) {
            attributes[a + j] = attributes[a + k];
        }
    }
  }

  static __device__ void round_multiplicities_to_ints_and_update_attributes(
    int64_t j,
    int64_t k,
    real_type nj,
    real_type nk,
    VectorView<real_type> attributes,
    VectorView<int64_t> multiplicity,
    real_type take_from_j,
    int64_t n_attr
  ) {
    multiplicity[j] = max((int64_t)(round(nj)), (int64_t)(1));
    multiplicity[k] = max((int64_t)(round(nk)), (int64_t)(1));
    auto factor_j = nj / multiplicity[j];
    auto factor_k = nk / multiplicity[k];
    for (auto a = 0; a < n_attr; a +=1) {
        attributes[a + k] *= factor_k;
        attributes[a + j] *= factor_j;
    }
  }

  static __device__ void break_up(
    int64_t i,
    int64_t j,
    int64_t k,
    int64_t cid,
    VectorView<int64_t> multiplicity,
    VectorView<real_type> gamma,
    VectorView<real_type> attributes,
    VectorView<real_type> fragment_mass,
    int64_t max_multiplicity,
    VectorView<int64_t> breakup_rate,
    VectorView<int64_t> breakup_rate_deficit,
    VectorView<real_type> particle_mass,
    int64_t n_sd,
    int64_t n_attr
  ) {
    real_type take_from_j[1] = {}; // float
    real_type new_mult_k[1] = {}; // float
    auto gamma_j_k = Commons::compute_transfer_multiplicities(
        gamma[i],
        j,
        k,
        multiplicity,
        particle_mass,
        fragment_mass[i],
        max_multiplicity,
        take_from_j,
        new_mult_k
    );
    auto gamma_deficit = gamma[i] - gamma_j_k;

    real_type nj[1] = {}; // float
    real_type nk[1] = {}; // float

    Commons::get_new_multiplicities_and_update_attributes(
        j, k, attributes, multiplicity, take_from_j[0], new_mult_k[0], n_attr, nj, nk
    );

    atomicAdd(
        (unsigned long long int*)&breakup_rate[cid],
        (unsigned long long int)(gamma_j_k * multiplicity[k])
    );
    atomicAdd(
        (unsigned long long int*)&breakup_rate_deficit[cid],
        (unsigned long long int)(gamma_deficit * multiplicity[k])
    );
    Commons::round_multiplicities_to_ints_and_update_attributes(
        j, k, nj[0], nk[0], attributes, multiplicity, take_from_j[0], n_attr
    );
  }
};
"""


class CollisionsMethods(
    ThrustRTCBackendMethods
):  # pylint: disable=too-many-instance-attributes
    @cached_property
    def __scale_prob_for_adaptive_sdm_gamma_body_1(self):
        return trtc.For(
            param_names=("dt_todo", "dt_left", "dt_max"),
            name_iter="cid",
            body="""
                dt_todo[cid] = min(dt_left[cid], dt_max);
            """,
        )

    @cached_property
    def __scale_prob_for_adaptive_sdm_gamma_body_2(self):
        return trtc.For(
            param_names=(
                "prob",
                "idx",
                "multiplicity",
                "cell_id",
                "dt",
                "is_first_in_pair",
                "dt_todo",
            ),
            name_iter="i",
            body=f"""
                {COMMONS}
                int64_t _j[1] = {{}};
                int64_t _k[1] = {{}};
                auto skip_pair = Commons::pair_indices(i, _j, _k, idx, is_first_in_pair, prob);
                auto j = _j[0];
                auto k = _k[0];
                if (skip_pair) {{
                    return;
                }}
                auto prop = (int64_t)(multiplicity[j] / multiplicity[k]);
                auto dt_optimal = dt * prop / prob[i];
                auto cid = cell_id[j];
                static_assert(sizeof(dt_todo[0]) == sizeof(unsigned int), "");
                atomicMin((unsigned int*)&dt_todo[cid], __float_as_uint(dt_optimal));
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __scale_prob_for_adaptive_sdm_gamma_body_3(self):
        return trtc.For(
            param_names=("prob", "idx", "cell_id", "dt", "is_first_in_pair", "dt_todo"),
            name_iter="i",
            body=f"""
                {COMMONS}
                int64_t _j[1] = {{}};
                int64_t _k[1] = {{}};
                auto skip_pair = Commons::pair_indices(i, _j, _k, idx, is_first_in_pair, prob);
                auto j = _j[0];
                auto k = _k[0];
                if (skip_pair) {{
                    return;
                }}
                prob[i] *= dt_todo[cell_id[j]] / dt;
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __scale_prob_for_adaptive_sdm_gamma_body_4(self):
        return trtc.For(
            param_names=("dt_left", "dt_todo", "stats_n_substep"),
            name_iter="cid",
            body="""
                dt_left[cid] -= dt_todo[cid];
                if (dt_todo[cid] > 0) {
                    stats_n_substep[cid] += 1;
                }
            """,
        )

    @cached_property
    def __sort_by_cell_id_and_update_cell_start_body(self):
        return trtc.For(
            param_names=("cell_id", "cell_start", "idx"),
            name_iter="i",
            body="""
            if (i == 0) {
                cell_start[cell_id[idx[0]]] = 0;
            } else {
                auto cell_id_curr = cell_id[idx[i]];
                auto cell_id_next = cell_id[idx[i + 1]];
                auto diff = (cell_id_next - cell_id_curr);
                for (auto j = 1; j < diff + 1; j += 1) {
                    cell_start[cell_id_curr + j] = idx[i + 1];
                }
            }
            """,
        )

    @cached_property
    def __collision_coalescence_body(self):
        return trtc.For(
            param_names=(
                "multiplicity",
                "idx",
                "n_sd",
                "attributes",
                "n_attr",
                "gamma",
                "healthy",
                "cell_id",
                "coalescence_rate",
                "is_first_in_pair",
            ),
            name_iter="i",
            body=f"""
            {COMMONS}
            int64_t _j[1] = {{}};
            int64_t _k[1] = {{}};
            auto skip_pair = Commons::pair_indices(i, _j, _k, idx, is_first_in_pair, gamma);
            auto j = _j[0];
            auto k = _k[0];
            if (skip_pair) {{
                return;
            }}

            Commons::coalesce(i, j, k, cell_id, multiplicity, gamma, attributes, coalescence_rate, n_attr, n_sd);

            Commons::flag_zero_multiplicity(j, k, multiplicity, healthy);
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __collision_coalescence_breakup_body(self):
        return trtc.For(
            param_names=(
                "multiplicity",
                "idx",
                "attributes",
                "gamma",
                "rand",
                "Ec",
                "Eb",
                "fragment_mass",
                "healthy",
                "cell_id",
                "coalescence_rate",
                "breakup_rate",
                "breakup_rate_deficit",
                "is_first_in_pair",
                "max_multiplicity",
                "particle_mass",
                "n_sd",
                "n_attr",
            ),
            name_iter="i",
            body=f"""
            {COMMONS}
            int64_t _j[1] = {{}};
            int64_t _k[1] = {{}};
            auto skip_pair = Commons::pair_indices(i, _j, _k, idx, is_first_in_pair, gamma);
            auto j = _j[0];
            auto k = _k[0];
            if (skip_pair) {{
                return;
            }}
            auto bouncing = rand[i] - (Ec[i] + (1 - Ec[i]) * (Eb[i])) > 0;
            if (bouncing) {{
                return;
            }}

            if (rand[i] - Ec[i] < 0) {{
                Commons::coalesce(
                    i,
                    j,
                    k,
                    cell_id,
                    multiplicity,
                    gamma,
                    attributes,
                    coalescence_rate,
                    n_attr,
                    n_sd
                );
            }} else {{
                Commons::break_up(
                    i, j, k,
                    cell_id[j],
                    multiplicity,
                    gamma,
                    attributes,
                    fragment_mass,
                    max_multiplicity,
                    breakup_rate,
                    breakup_rate_deficit,
                    particle_mass,
                    n_sd,
                    n_attr
                );
            }}

            Commons::flag_zero_multiplicity(j, k, multiplicity, healthy);
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __compute_gamma_body(self):
        return trtc.For(
            param_names=(
                "prob",
                "rand",
                "idx",
                "multiplicity",
                "cell_id",
                "collision_rate_deficit",
                "collision_rate",
                "is_first_in_pair",
                "out",
            ),
            name_iter="i",
            body=f"""
            {COMMONS}
            out[i] = ceil(prob[i] - rand[i]);

            int64_t _j[1] = {{}};
            int64_t _k[1] = {{}};
            auto skip_pair = Commons::pair_indices(i, _j, _k, idx, is_first_in_pair, out);
            auto j = _j[0];
            auto k = _k[0];
            if (skip_pair) {{
                return;
            }}

            auto prop = (int64_t)(multiplicity[j] / multiplicity[k]);
            auto g = min((int64_t)(out[i]), prop);

            atomicAdd(
                (unsigned long long int*)&collision_rate[cell_id[j]],
                (unsigned long long int)(g * multiplicity[k])
            );
            atomicAdd(
                (unsigned long long int*)&collision_rate_deficit[cell_id[j]],
                (unsigned long long int)(((int64_t)(out[i]) - g) * multiplicity[k])
            );

            out[i] = g;
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __normalize_body_0(self):
        return trtc.For(
            param_names=("cell_start", "norm_factor", "dt_div_dv"),
            name_iter="i",
            body="""
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

    @cached_property
    def __normalize_body_1(self):
        return trtc.For(
            param_names=("prob", "cell_id", "norm_factor"),
            name_iter="i",
            body="""
            prob[i] *= norm_factor[cell_id[i]];
            """,
        )

    @cached_property
    def __remove_zero_n_or_flagged_body(self):
        return trtc.For(
            param_names=("data", "idx", "n_sd"),
            name_iter="i",
            body="""
            if (idx[i] < n_sd && data[idx[i]] == 0) {
                idx[i] = n_sd;
            }
            """,
        )

    @cached_property
    def __cell_id_body(self):
        return trtc.For(
            param_names=("cell_id", "cell_origin", "strides", "n_dims", "size"),
            name_iter="i",
            body="""
            cell_id[i] = 0;
            for (auto j = 0; j < n_dims; j += 1) {
                cell_id[i] += cell_origin[size * j + i] * strides[j];
            }
            """,
        )

    @cached_property
    def __exp_fragmentation_body(self):
        return trtc.For(
            param_names=("scale", "frag_volume", "rand", "tol"),
            name_iter="i",
            body="""
            frag_volume[i] = -scale * log(max(1 - rand[i], tol));
            """,
        )

    @cached_property
    def __fragmentation_limiters_body(self):
        return trtc.For(
            param_names=(
                "n_fragment",
                "frag_volume",
                "x_plus_y",
                "vmin",
                "nfmax",
                "nfmax_is_not_none",
            ),
            name_iter="i",
            body="""
            frag_volume[i] = min(frag_volume[i], x_plus_y[i]);

            if (nfmax_is_not_none) {
                if (x_plus_y[i] / frag_volume[i] > nfmax) {
                    frag_volume[i] = x_plus_y[i] / nfmax;
                }
            }
            else if (frag_volume[i] < vmin) {
                frag_volume[i] = x_plus_y[i];
            }
            else if (frag_volume[i] == 0.0) {
                frag_volume[i] = x_plus_y[i];
            }
            n_fragment[i] = x_plus_y[i] / frag_volume[i];
            """,
        )

    @cached_property
    def __gauss_fragmentation_body(self):
        return trtc.For(
            param_names=("mu", "sigma", "frag_volume", "rand"),
            name_iter="i",
            body=f"""
            frag_volume[i] = mu + sigma * {self.formulae.trivia.erfinv_approx.c_inline(
                c="rand[i]"
            )};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __slams_fragmentation_body(self):
        return trtc.For(
            param_names=("n_fragment", "frag_volume", "x_plus_y", "probs", "rand"),
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
            frag_volume[i] = x_plus_y[i] / n_fragment[i];
            """,
        )

    @cached_property
    def __feingold1988_fragmentation_body(self):
        return trtc.For(
            param_names=("scale", "frag_volume", "x_plus_y", "rand", "fragtol"),
            name_iter="i",
            body=f"""
            frag_volume[i] = {self.formulae.fragmentation_function.frag_volume.c_inline(
                scale="scale",
                rand="rand[i]",
                x_plus_y="x_plus_y[i]",
                fragtol="fragtol"
            )};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __straub_Nr_body(self):
        return """
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

    @cached_property
    def __straub_mass_remainder(self):
        const = self.formulae.constants
        return f"""
                Nr1[i] = Nr1[i] * exp(3 * mu1 + 9 * pow(sigma1, 2) / 2);
                Nr2[i] = Nr2[i] * (pow(mu2, 3) + 3 * mu2 * pow(sigma2, 2));
                Nr3[i] = Nr3[i] * (pow(mu3, 3) + 3 * mu3 * pow(sigma3, 2));
                Nr4[i] = v_max[i] * 6 / {const.PI} + pow(ds[i], 3) - Nr1[i] - Nr2[i] - Nr3[i];
                if (Nr4[i] <= 0.0) {{
                    d34[i] = 0;
                    Nr4[i] = 0;
                }}
                else {{
                    d34[i] = exp(log(Nr4[i]) / 3);
                }}
        """

    @cached_property
    def __straub_fragmentation_body(self):
        const = self.formulae.constants
        return trtc.For(
            param_names=(
                "CW",
                "gam",
                "ds",
                "frag_volume",
                "v_max",
                "rand",
                "Nr1",
                "Nr2",
                "Nr3",
                "Nr4",
                "Nrt",
                "d34",
            ),
            name_iter="i",
            body=f"""
                {self.__straub_Nr_body}
                auto sigma1 = {self.formulae.fragmentation_function.params_sigma1.c_inline(CW="CW[i]")};
                auto mu1 = {self.formulae.fragmentation_function.params_mu1.c_inline(sigma1="sigma1")};
                auto sigma2 = {self.formulae.fragmentation_function.params_sigma2.c_inline(CW="CW[i]")};
                auto mu2 = {self.formulae.fragmentation_function.params_mu2.c_inline(ds="ds[i]")};
                auto sigma3 = {self.formulae.fragmentation_function.params_sigma3.c_inline(CW="CW[i]")};
                auto mu3 = {self.formulae.fragmentation_function.params_mu3.c_inline(ds="ds[i]")};
                {self.__straub_mass_remainder}
                Nrt[i] = Nr1[i] + Nr2[i] + Nr3[i] + Nr4[i];

                if (rand[i] < Nr1[i] / Nrt[i]) {{
                    auto X = rand[i] * Nrt[i] / Nr1[i];
                    auto lnarg = mu1 + sqrt(2.0) * sigma1 * {self.formulae.trivia.erfinv_approx.c_inline(
                        c="X"
                    )};
                    frag_volume[i] = exp(lnarg);
                }}
                else if (rand[i] < (Nr2[i] + Nr1[i]) / Nrt[i]) {{
                    auto X = (rand[i] * Nrt[i] - Nr1[i]) / Nr2[i];
                    frag_volume[i] = mu2 + sqrt(2.0) * sigma2 * {self.formulae.trivia.erfinv_approx.c_inline(
                        c="X"
                    )};
                }}
                else if (rand[i] < (Nr3[i] + Nr2[i] + Nr1[i]) / Nrt[i]) {{
                    auto X = (rand[i] * Nrt[i] - Nr1[i] - Nr2[i]) / Nr3[i];
                    frag_volume[i] = mu3 + sqrt(2.0) * sigma3 * {self.formulae.trivia.erfinv_approx.c_inline(
                        c="X"
                    )};
                }}
                else {{
                    frag_volume[i] = d34[i];
                }}

                frag_volume[i] = pow(frag_volume[i], 3) * {const.PI} / 6;
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def adaptive_sdm_end(self, dt_left, cell_start):
        i = trtc.Find(dt_left.data, self._get_floating_point(0))
        if i is None:
            i = len(dt_left)
        return cell_start[i]

    # pylint: disable=unused-argument
    @nice_thrust(**NICE_THRUST_FLAGS)
    def scale_prob_for_adaptive_sdm_gamma(
        self,
        *,
        prob,
        multiplicity,
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

        self.__scale_prob_for_adaptive_sdm_gamma_body_1.launch_n(
            n=len(dt_left), args=(dt_todo, dt_left.data, d_dt_max)
        )
        self.__scale_prob_for_adaptive_sdm_gamma_body_2.launch_n(
            n=len(multiplicity) // 2,
            args=(
                prob.data,
                multiplicity.idx.data,
                multiplicity.data,
                cell_id.data,
                d_dt,
                is_first_in_pair.indicator.data,
                dt_todo,
            ),
        )
        self.__scale_prob_for_adaptive_sdm_gamma_body_3.launch_n(
            n=len(multiplicity) // 2,
            args=(
                prob.data,
                multiplicity.idx.data,
                cell_id.data,
                d_dt,
                is_first_in_pair.indicator.data,
                dt_todo,
            ),
        )
        self.__scale_prob_for_adaptive_sdm_gamma_body_4.launch_n(
            n=len(dt_left), args=(dt_left.data, dt_todo, stats_n_substep.data)
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
            n=len(cell_id),
            args=(cell_id.data, cell_origin.data, strides.data, n_dims, size),
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
            n=len(idx) // 2,
            args=(
                multiplicity.data,
                idx.data,
                n_sd,
                attributes.data,
                n_attr,
                gamma.data,
                healthy.data,
                cell_id.data,
                coalescence_rate.data,
                is_first_in_pair.indicator.data,
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def collision_coalescence_breakup(  # pylint: disable=unused-argument,too-many-locals
        self,
        *,
        multiplicity,
        idx,
        attributes,
        gamma,
        rand,
        Ec,
        Eb,
        fragment_mass,
        healthy,
        cell_id,
        coalescence_rate,
        breakup_rate,
        breakup_rate_deficit,
        is_first_in_pair,
        warn_overflows,
        particle_mass,
        max_multiplicity,
    ):
        if warn_overflows:
            raise NotImplementedError()
        if len(idx) < 2:
            return
        n_sd = trtc.DVInt64(attributes.shape[1])
        n_attr = trtc.DVInt64(attributes.shape[0])
        self.__collision_coalescence_breakup_body.launch_n(
            n=len(idx) // 2,
            args=(
                multiplicity.data,
                idx.data,
                attributes.data,
                gamma.data,
                rand.data,
                Ec.data,
                Eb.data,
                fragment_mass.data,
                healthy.data,
                cell_id.data,
                coalescence_rate.data,
                breakup_rate.data,
                breakup_rate_deficit.data,
                is_first_in_pair.indicator.data,
                trtc.DVInt64(max_multiplicity),
                particle_mass.data,
                n_sd,
                n_attr,
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def compute_gamma(
        self,
        *,
        prob,
        rand,
        multiplicity,
        cell_id,
        collision_rate_deficit,
        collision_rate,
        is_first_in_pair,
        out,
    ):
        if len(multiplicity) < 2:
            return
        self.__compute_gamma_body.launch_n(
            n=len(multiplicity) // 2,
            args=(
                prob.data,
                rand.data,
                multiplicity.idx.data,
                multiplicity.data,
                cell_id.data,
                collision_rate_deficit.data,
                collision_rate.data,
                is_first_in_pair.indicator.data,
                out.data,
            ),
        )

    # pylint: disable=unused-argument
    def make_cell_caretaker(self, idx_shape, idx_dtype, cell_start_len, scheme=None):
        if not (scheme is None or scheme == "default"):
            raise NotImplementedError()
        return self._sort_by_cell_id_and_update_cell_start

    # pylint: disable=unused-argument
    @nice_thrust(**NICE_THRUST_FLAGS)
    def normalize(
        self, *, prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv
    ):
        n_cell = cell_start.shape[0] - 1
        device_dt_div_dv = self._get_floating_point(timestep / dv)
        self.__normalize_body_0.launch_n(
            n=n_cell, args=(cell_start.data, norm_factor.data, device_dt_div_dv)
        )
        self.__normalize_body_1.launch_n(
            prob.shape[0], (prob.data, cell_id.data, norm_factor.data)
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def remove_zero_n_or_flagged(self, data, idx, length) -> int:
        n_sd = trtc.DVInt64(idx.size())

        # Warning: (potential bug source): reading from outside of array
        self.__remove_zero_n_or_flagged_body.launch_n(n=length, args=(data, idx, n_sd))

        trtc.Sort(idx)

        new_length = idx.size() - trtc.Count(idx, n_sd)
        return new_length

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
        #   no handling of cell_idx in __sort_by_cell_id_and_update_cell_start_body yet
        trtc.Fill(cell_start.data, trtc.DVInt64(len(idx)))
        if len(idx) > 1:
            self.__sort_by_cell_id_and_update_cell_start_body.launch_n(
                n=len(idx) - 1, args=(cell_id.data, cell_start.data, idx.data)
            )

    def exp_fragmentation(
        self,
        *,
        n_fragment,
        scale,
        frag_volume,
        x_plus_y,
        rand,
        vmin,
        nfmax,
        tol=1e-5,
    ):
        self.__exp_fragmentation_body.launch_n(
            n=len(frag_volume),
            args=(
                self._get_floating_point(scale),
                frag_volume.data,
                rand.data,
                self._get_floating_point(tol),
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_volume),
            args=(
                n_fragment.data,
                frag_volume.data,
                x_plus_y.data,
                self._get_floating_point(vmin),
                self._get_floating_point(nfmax if nfmax else -1),
                trtc.DVBool(nfmax is not None),
            ),
        )

    def gauss_fragmentation(
        self, *, n_fragment, mu, sigma, frag_volume, x_plus_y, rand, vmin, nfmax
    ):
        self.__gauss_fragmentation_body.launch_n(
            n=len(frag_volume),
            args=(
                self._get_floating_point(mu),
                self._get_floating_point(sigma),
                frag_volume.data,
                rand.data,
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_volume),
            args=(
                n_fragment.data,
                frag_volume.data,
                x_plus_y.data,
                self._get_floating_point(vmin),
                self._get_floating_point(nfmax if nfmax else -1),
                trtc.DVBool(nfmax is not None),
            ),
        )

    def slams_fragmentation(
        self, n_fragment, frag_volume, x_plus_y, probs, rand, vmin, nfmax
    ):  # pylint: disable=too-many-arguments
        self.__slams_fragmentation_body.launch_n(
            n=(len(n_fragment)),
            args=(
                n_fragment.data,
                frag_volume.data,
                x_plus_y.data,
                probs.data,
                rand.data,
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_volume),
            args=(
                n_fragment.data,
                frag_volume.data,
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
        frag_volume,
        x_plus_y,
        rand,
        fragtol,
        vmin,
        nfmax,
    ):
        self.__feingold1988_fragmentation_body.launch_n(
            n=len(frag_volume),
            args=(
                self._get_floating_point(scale),
                frag_volume.data,
                x_plus_y.data,
                rand.data,
                self._get_floating_point(fragtol),
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_volume),
            args=(
                n_fragment.data,
                frag_volume.data,
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
        frag_volume,
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
        d34,
    ):
        self.__straub_fragmentation_body.launch_n(
            n=len(frag_volume),
            args=(
                CW.data,
                gam.data,
                ds.data,
                frag_volume.data,
                v_max.data,
                rand.data,
                Nr1.data,
                Nr2.data,
                Nr3.data,
                Nr4.data,
                Nrt.data,
                d34.data,
            ),
        )

        self.__fragmentation_limiters_body.launch_n(
            n=len(frag_volume),
            args=(
                n_fragment.data,
                frag_volume.data,
                x_plus_y.data,
                self._get_floating_point(vmin),
                self._get_floating_point(nfmax if nfmax else -1),
                trtc.DVBool(nfmax is not None),
            ),
        )
