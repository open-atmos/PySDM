"""
CPU implementation of backend methods for particle collisions
"""

# pylint: disable=too-many-lines
import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_numba.atomic_operations import atomic_add
from PySDM.backends.impl_numba.storage import Storage
from PySDM.backends.impl_numba.warnings import warn


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def pair_indices(i, idx, is_first_in_pair, prob_like):
    """given permutation array `idx` and `is_first_in_pair` flag array,
    returns indices `j` and `k` of droplets within pair `i` and a `skip_pair` flag,
    `j` points to the droplet that is first in pair (higher or equal multiplicity)
    output is valid only if `2*i` or `2*i+1` points to a valid pair start index (within one cell)
    otherwise the `skip_pair` flag is set to True and returned `j` & `k` indices are set to -1.
    In addition, the `prob_like` array is checked for zeros at position `i`, in which case
    the `skip_pair` is also set to `True`
    """
    skip_pair = False

    if prob_like[i] == 0:
        skip_pair = True
        j, k = -1, -1
    else:
        offset = 1 - is_first_in_pair[2 * i]
        j = idx[2 * i + offset]
        k = idx[2 * i + 1 + offset]
    return j, k, skip_pair


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def flag_zero_multiplicity(j, k, multiplicity, healthy):
    if multiplicity[k] == 0 or multiplicity[j] == 0:
        healthy[0] = 0


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def coalesce(  # pylint: disable=too-many-arguments
    i, j, k, cid, multiplicity, gamma, attributes, coalescence_rate
):
    atomic_add(coalescence_rate, cid, gamma[i] * multiplicity[k])
    new_n = multiplicity[j] - gamma[i] * multiplicity[k]
    if new_n > 0:
        multiplicity[j] = new_n
        for a in range(len(attributes)):
            attributes[a, k] += gamma[i] * attributes[a, j]
    else:  # new_n == 0
        multiplicity[j] = multiplicity[k] // 2
        multiplicity[k] = multiplicity[k] - multiplicity[j]
        for a in range(len(attributes)):
            attributes[a, j] = gamma[i] * attributes[a, j] + attributes[a, k]
            attributes[a, k] = attributes[a, j]


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def compute_transfer_multiplicities(
    gamma, j, k, multiplicity, particle_mass, fragment_mass_i, max_multiplicity
):  # pylint: disable=too-many-arguments
    overflow_flag = False
    gamma_j_k = 0
    take_from_j_test = multiplicity[k]
    take_from_j = 0
    new_mult_k_test = (
        (particle_mass[j] + particle_mass[k]) / fragment_mass_i
    ) * multiplicity[k]
    new_mult_k = multiplicity[k]
    for m in range(int(gamma)):
        # check for overflow of multiplicity
        if new_mult_k_test > max_multiplicity:
            overflow_flag = True
            break

        # check for new_n >= 0
        if take_from_j_test > multiplicity[j]:
            break

        take_from_j = take_from_j_test
        new_mult_k = new_mult_k_test
        gamma_j_k = m + 1

        take_from_j_test += new_mult_k_test
        new_mult_k_test = (
            new_mult_k_test * (particle_mass[j] / fragment_mass_i) + new_mult_k_test
        )

    return take_from_j, new_mult_k, gamma_j_k, overflow_flag


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def get_new_multiplicities_and_update_attributes(
    j, k, attributes, multiplicity, take_from_j, new_mult_k
):  # pylint: disable=too-many-arguments
    for a in range(len(attributes)):
        attributes[a, k] *= multiplicity[k]
        attributes[a, k] += take_from_j * attributes[a, j]
        attributes[a, k] /= new_mult_k

    if multiplicity[j] > take_from_j:
        nj = multiplicity[j] - take_from_j
        nk = new_mult_k

    else:  # take_from_j == multiplicity[j]
        nj = new_mult_k / 2
        nk = nj
        for a in range(len(attributes)):
            attributes[a, j] = attributes[a, k]
    return nj, nk


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def round_multiplicities_to_ints_and_update_attributes(
    j,
    k,
    nj,
    nk,
    attributes,
    multiplicity,
):  # pylint: disable=too-many-arguments
    multiplicity[j] = max(round(nj), 1)
    multiplicity[k] = max(round(nk), 1)
    factor_j = nj / multiplicity[j]
    factor_k = nk / multiplicity[k]
    for a in range(len(attributes)):
        attributes[a, k] *= factor_k
        attributes[a, j] *= factor_j


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def break_up(  # pylint: disable=too-many-arguments,c,too-many-locals
    i,
    j,
    k,
    cid,
    multiplicity,
    gamma,
    attributes,
    fragment_mass,
    max_multiplicity,
    breakup_rate,
    breakup_rate_deficit,
    warn_overflows,
    particle_mass,
):  # breakup0 guarantees take_from_j <= multiplicity[j]
    take_from_j, new_mult_k, gamma_j_k, overflow_flag = compute_transfer_multiplicities(
        gamma[i],
        j,
        k,
        multiplicity,
        particle_mass,
        fragment_mass[i],
        max_multiplicity,
    )
    gamma_deficit = gamma[i] - gamma_j_k

    # breakup1 also handles new_n[j] == 0 case via splitting
    nj, nk = get_new_multiplicities_and_update_attributes(
        j, k, attributes, multiplicity, take_from_j, new_mult_k
    )

    atomic_add(breakup_rate, cid, gamma_j_k * multiplicity[k])
    atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])

    # breakup2 also guarantees that no multiplicities are set to 0
    round_multiplicities_to_ints_and_update_attributes(
        j, k, nj, nk, attributes, multiplicity
    )
    if overflow_flag and warn_overflows:
        warn("overflow", __file__)


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def break_up_while(
    i,
    j,
    k,
    cid,
    multiplicity,
    gamma,
    attributes,
    fragment_mass,
    max_multiplicity,
    breakup_rate,
    breakup_rate_deficit,
    warn_overflows,
    particle_mass,
):  # pylint: disable=too-many-arguments,unused-argument,too-many-locals
    gamma_deficit = gamma[i]
    overflow_flag = False
    while gamma_deficit > 0:
        if multiplicity[k] == multiplicity[j]:
            take_from_j = multiplicity[j]
            new_mult_k = (
                (particle_mass[j] + particle_mass[k])
                / fragment_mass[i]
                * multiplicity[k]
            )

            # check for overflow
            if new_mult_k > max_multiplicity:
                atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])
                overflow_flag = True
                break
            gamma_j_k = gamma_deficit

        else:
            if multiplicity[k] > multiplicity[j]:
                j, k = k, j
            (
                take_from_j,
                new_mult_k,
                gamma_j_k,
                overflow_flag,
            ) = compute_transfer_multiplicities(
                gamma_deficit,
                j,
                k,
                multiplicity,
                particle_mass,
                fragment_mass[i],
                max_multiplicity,
            )

        nj, nk = get_new_multiplicities_and_update_attributes(
            j, k, attributes, multiplicity, take_from_j, new_mult_k
        )

        atomic_add(breakup_rate, cid, gamma_j_k * multiplicity[k])
        gamma_deficit -= gamma_j_k
        round_multiplicities_to_ints_and_update_attributes(
            j, k, nj, nk, attributes, multiplicity
        )

    atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])

    if overflow_flag and warn_overflows:
        warn("overflow", __file__)


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def straub_Nr(  # pylint: disable=too-many-arguments,unused-argument
    i,
    Nr1,
    Nr2,
    Nr3,
    Nr4,
    Nrt,
    CW,
    gam,
):  # pylint: disable=too-many-branches`
    if gam[i] * CW[i] >= 7.0:
        Nr1[i] = 0.088 * (gam[i] * CW[i] - 7.0)
    if CW[i] >= 21.0:
        Nr2[i] = 0.22 * (CW[i] - 21.0)
        if CW[i] <= 46.0:
            Nr3[i] = 0.04 * (46.0 - CW[i])
    else:
        Nr3[i] = 1.0
    Nr4[i] = 1.0
    Nrt[i] = Nr1[i] + Nr2[i] + Nr3[i] + Nr4[i]


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def straub_mass_remainder(  # pylint: disable=too-many-arguments,unused-argument
    i, vl, ds, mu1, sigma1, mu2, sigma2, mu3, sigma3, d34, Nr1, Nr2, Nr3, Nr4
):
    # pylint: disable=too-many-arguments, too-many-locals
    Nr1[i] = Nr1[i] * np.exp(3 * mu1 + 9 * np.power(sigma1, 2) / 2)
    Nr2[i] = Nr2[i] * (mu2**3 + 3 * mu2 * sigma2**2)
    Nr3[i] = Nr3[i] * (mu3**3 + 3 * mu3 * sigma3**2)
    Nr4[i] = vl[i] * 6 / np.pi + ds[i] ** 3 - Nr1[i] - Nr2[i] - Nr3[i]
    if Nr4[i] <= 0.0:
        d34[i] = 0
        Nr4[i] = 0
    else:
        d34[i] = np.exp(np.log(Nr4[i]) / 3)


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def ll82_Nr(  # pylint: disable=too-many-arguments,unused-argument
    i,
    Rf,
    Rs,
    Rd,
    CKE,
    W,
    W2,
):  # pylint: disable=too-many-branches`
    if CKE[i] >= 0.893e-6:
        Rf[i] = 1.11e-4 * CKE[i] ** (-0.654)
    else:
        Rf[i] = 1.0
    if W[i] >= 0.86:
        Rs[i] = 0.685 * (1 - np.exp(-1.63 * (W2[i] - 0.86)))
    else:
        Rs[i] = 0.0
    if (Rs[i] + Rf[i]) > 1.0:
        Rd[i] = 0.0
    else:
        Rd[i] = 1.0 - Rs[i] - Rf[i]


class CollisionsMethods(BackendMethods):
    def __init__(self):  # pylint: disable=too-many-statements,too-many-locals
        BackendMethods.__init__(self)

        _break_up = break_up_while if self.formulae.handle_all_breakups else break_up
        const = self.formulae.constants

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def __collision_coalescence_breakup_body(
            *,
            multiplicity,
            idx,
            length,
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
            max_multiplicity,
            warn_overflows,
            particle_mass,
        ):
            # pylint: disable=not-an-iterable,too-many-nested-blocks,too-many-locals
            for i in numba.prange(length // 2):
                j, k, skip_pair = pair_indices(i, idx, is_first_in_pair, gamma)
                if skip_pair:
                    continue
                bouncing = rand[i] - (Ec[i] + (1 - Ec[i]) * (Eb[i])) > 0
                if bouncing:
                    continue

                if rand[i] - Ec[i] < 0:
                    coalesce(
                        i,
                        j,
                        k,
                        cell_id[j],
                        multiplicity,
                        gamma,
                        attributes,
                        coalescence_rate,
                    )
                else:
                    _break_up(
                        i,
                        j,
                        k,
                        cell_id[j],
                        multiplicity,
                        gamma,
                        attributes,
                        fragment_mass,
                        max_multiplicity,
                        breakup_rate,
                        breakup_rate_deficit,
                        warn_overflows,
                        particle_mass,
                    )
                flag_zero_multiplicity(j, k, multiplicity, healthy)

        self.__collision_coalescence_breakup_body = __collision_coalescence_breakup_body

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def __ll82_coalescence_check_body(*, Ec, dl):
            for i in numba.prange(len(Ec)):  # pylint: disable=not-an-iterable
                if dl[i] < 0.4e-3:
                    Ec[i] = 1.0

        self.__ll82_coalescence_check_body = __ll82_coalescence_check_body

        if self.formulae.fragmentation_function.__name__ == "Straub2010Nf":
            straub_sigma1 = self.formulae.fragmentation_function.params_sigma1
            straub_mu1 = self.formulae.fragmentation_function.params_mu1
            straub_sigma2 = self.formulae.fragmentation_function.params_sigma2
            straub_mu2 = self.formulae.fragmentation_function.params_mu2
            straub_sigma3 = self.formulae.fragmentation_function.params_sigma3
            straub_mu3 = self.formulae.fragmentation_function.params_mu3
            straub_erfinv = self.formulae.trivia.erfinv_approx

            @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
            def __straub_fragmentation_body(
                *, CW, gam, ds, v_max, frag_volume, rand, Nr1, Nr2, Nr3, Nr4, Nrt, d34
            ):  # pylint: disable=too-many-arguments,too-many-locals
                for i in numba.prange(  # pylint: disable=not-an-iterable
                    len(frag_volume)
                ):
                    straub_Nr(i, Nr1, Nr2, Nr3, Nr4, Nrt, CW, gam)
                    sigma1 = straub_sigma1(CW[i])
                    mu1 = straub_mu1(sigma1)
                    sigma2 = straub_sigma2(CW[i])
                    mu2 = straub_mu2(ds[i])
                    sigma3 = straub_sigma3(CW[i])
                    mu3 = straub_mu3(ds[i])
                    straub_mass_remainder(
                        i,
                        v_max,
                        ds,
                        mu1,
                        sigma1,
                        mu2,
                        sigma2,
                        mu3,
                        sigma3,
                        d34,
                        Nr1,
                        Nr2,
                        Nr3,
                        Nr4,
                    )
                    Nrt[i] = Nr1[i] + Nr2[i] + Nr3[i] + Nr4[i]

                    if Nrt[i] == 0.0:
                        diameter = 0.0
                    else:
                        if rand[i] < Nr1[i] / Nrt[i]:
                            X = rand[i] * Nrt[i] / Nr1[i]
                            lnarg = mu1 + np.sqrt(2) * sigma1 * straub_erfinv(X)
                            diameter = np.exp(lnarg)
                        elif rand[i] < (Nr2[i] + Nr1[i]) / Nrt[i]:
                            X = (rand[i] * Nrt[i] - Nr1[i]) / Nr2[i]
                            diameter = mu2 + np.sqrt(2) * sigma2 * straub_erfinv(X)
                        elif rand[i] < (Nr3[i] + Nr2[i] + Nr1[i]) / Nrt[i]:
                            X = (rand[i] * Nrt[i] - Nr1[i] - Nr2[i]) / Nr3[i]
                            diameter = mu3 + np.sqrt(2) * sigma3 * straub_erfinv(X)
                        else:
                            diameter = d34[i]

                    frag_volume[i] = diameter**3 * const.PI / 6

            self.__straub_fragmentation_body = __straub_fragmentation_body
        elif self.formulae.fragmentation_function.__name__ == "LowList1982Nf":
            ll82_params_f1 = self.formulae.fragmentation_function.params_f1
            ll82_params_f2 = self.formulae.fragmentation_function.params_f2
            ll82_params_f3 = self.formulae.fragmentation_function.params_f3
            ll82_params_s1 = self.formulae.fragmentation_function.params_s1
            ll82_params_s2 = self.formulae.fragmentation_function.params_s2
            ll82_params_d1 = self.formulae.fragmentation_function.params_d1
            ll82_params_d2 = self.formulae.fragmentation_function.params_d2
            ll82_erfinv = self.formulae.fragmentation_function.erfinv

            @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
            def __ll82_fragmentation_body(
                *, CKE, W, W2, St, ds, dl, dcoal, frag_volume, rand, Rf, Rs, Rd, tol
            ):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
                for i in numba.prange(  # pylint: disable=not-an-iterable
                    len(frag_volume)
                ):
                    if dl[i] <= 0.4e-3:
                        frag_volume[i] = dcoal[i] ** 3 * const.PI / 6
                    elif ds[i] == 0.0 or dl[i] == 0.0:
                        frag_volume[i] = 1e-18
                    else:
                        ll82_Nr(i, Rf, Rs, Rd, CKE, W, W2)
                        if rand[i] <= Rf[i]:  # filament breakup
                            (H1, mu1, sigma1) = ll82_params_f1(dl[i], dcoal[i])
                            (H2, mu2, sigma2) = ll82_params_f2(ds[i])
                            (H3, mu3, sigma3) = ll82_params_f3(ds[i], dl[i])
                            H1 = H1 * mu1
                            H2 = H2 * mu2
                            H3 = H3 * np.exp(mu3)
                            Hsum = H1 + H2 + H3
                            rand[i] = rand[i] / Rf[i]
                            if rand[i] <= H1 / Hsum:
                                X = max(rand[i] * Hsum / H1, tol)
                                frag_volume[i] = mu1 + np.sqrt(
                                    2
                                ) * sigma1 * ll82_erfinv(2 * X - 1)
                            elif rand[i] <= (H1 + H2) / Hsum:
                                X = (rand[i] * Hsum - H1) / H2
                                frag_volume[i] = mu2 + np.sqrt(
                                    2
                                ) * sigma2 * ll82_erfinv(2 * X - 1)
                            else:
                                X = min((rand[i] * Hsum - H1 - H2) / H3, 1.0 - tol)
                                lnarg = mu3 + np.sqrt(2) * sigma3 * ll82_erfinv(
                                    2 * X - 1
                                )
                                frag_volume[i] = np.exp(lnarg)

                        elif rand[i] <= Rf[i] + Rs[i]:  # sheet breakup
                            (H1, mu1, sigma1) = ll82_params_s1(dl[i], ds[i], dcoal[i])
                            (H2, mu2, sigma2) = ll82_params_s2(dl[i], ds[i], St[i])
                            H1 = H1 * mu1
                            H2 = H2 * np.exp(mu2)
                            Hsum = H1 + H2
                            rand[i] = (rand[i] - Rf[i]) / (Rs[i])
                            if rand[i] <= H1 / Hsum:
                                X = max(rand[i] * Hsum / H1, tol)
                                frag_volume[i] = mu1 + np.sqrt(
                                    2
                                ) * sigma1 * ll82_erfinv(2 * X - 1)
                            else:
                                X = min((rand[i] * Hsum - H1) / H2, 1.0 - tol)
                                lnarg = mu2 + np.sqrt(2) * sigma2 * ll82_erfinv(
                                    2 * X - 1
                                )
                                frag_volume[i] = np.exp(lnarg)

                        else:  # disk breakup
                            (H1, mu1, sigma1) = ll82_params_d1(
                                W[i], dl[i], dcoal[i], CKE[i]
                            )
                            (H2, mu2, sigma2) = ll82_params_d2(ds[i], dl[i], CKE[i])
                            H1 = H1 * mu1
                            Hsum = H1 + H2
                            rand[i] = (rand[i] - Rf[i] - Rs[i]) / Rd[i]
                            if rand[i] <= H1 / Hsum:
                                X = max(rand[i] * Hsum / H1, tol)
                                frag_volume[i] = mu1 + np.sqrt(
                                    2
                                ) * sigma1 * ll82_erfinv(2 * X - 1)
                            else:
                                X = min((rand[i] * Hsum - H1) / H2, 1 - tol)
                                lnarg = mu2 + np.sqrt(2) * sigma2 * ll82_erfinv(
                                    2 * X - 1
                                )
                                frag_volume[i] = np.exp(lnarg)

                        frag_volume[i] = (
                            frag_volume[i] * 0.01
                        )  # diameter in cm; convert to m
                        frag_volume[i] = frag_volume[i] ** 3 * const.PI / 6

            self.__ll82_fragmentation_body = __ll82_fragmentation_body
        elif self.formulae.fragmentation_function.__name__ == "Gaussian":
            erfinv_approx = self.formulae.trivia.erfinv_approx

            @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
            def __gauss_fragmentation_body(
                *, mu, sigma, frag_volume, rand
            ):  # pylint: disable=too-many-arguments
                for i in numba.prange(  # pylint: disable=not-an-iterable
                    len(frag_volume)
                ):
                    frag_volume[i] = mu + sigma * erfinv_approx(rand[i])

            self.__gauss_fragmentation_body = __gauss_fragmentation_body
        elif self.formulae.fragmentation_function.__name__ == "Feingold1988":
            feingold1988_frag_volume = self.formulae.fragmentation_function.frag_volume

            @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
            # pylint: disable=too-many-arguments
            def __feingold1988_fragmentation_body(
                *, scale, frag_volume, x_plus_y, rand, fragtol
            ):
                for i in numba.prange(  # pylint: disable=not-an-iterable
                    len(frag_volume)
                ):
                    frag_volume[i] = feingold1988_frag_volume(
                        scale, rand[i], x_plus_y[i], fragtol
                    )

            self.__feingold1988_fragmentation_body = __feingold1988_fragmentation_body

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
    def __adaptive_sdm_end_body(dt_left, n_cell, cell_start):
        end = 0
        for i in range(n_cell - 1, -1, -1):
            if dt_left[i] == 0:
                continue
            end = cell_start[i + 1]
            break
        return end

    def adaptive_sdm_end(self, dt_left, cell_start):
        return self.__adaptive_sdm_end_body(dt_left.data, len(dt_left), cell_start.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments,too-many-locals
    def __scale_prob_for_adaptive_sdm_gamma_body(
        prob,
        idx,
        length,
        multiplicity,
        cell_id,
        dt_left,
        dt,
        dt_range,
        is_first_in_pair,
        stats_n_substep,
        stats_dt_min,
    ):
        dt_todo = np.empty_like(dt_left)
        for cid in numba.prange(len(dt_todo)):  # pylint: disable=not-an-iterable
            dt_todo[cid] = min(dt_left[cid], dt_range[1])
        for i in range(length // 2):  # TODO #571
            j, k, skip_pair = pair_indices(i, idx, is_first_in_pair, prob)
            if skip_pair:
                continue
            prop = multiplicity[j] // multiplicity[k]
            dt_optimal = dt * prop / prob[i]
            cid = cell_id[j]
            dt_optimal = max(dt_optimal, dt_range[0])
            dt_todo[cid] = min(dt_todo[cid], dt_optimal)
            stats_dt_min[cid] = min(stats_dt_min[cid], dt_optimal)
        for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
            j, _, skip_pair = pair_indices(i, idx, is_first_in_pair, prob)
            if skip_pair:
                continue
            prob[i] *= dt_todo[cell_id[j]] / dt
        for cid in numba.prange(len(dt_todo)):  # pylint: disable=not-an-iterable
            dt_left[cid] -= dt_todo[cid]
            if dt_todo[cid] > 0:
                stats_n_substep[cid] += 1

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
        return self.__scale_prob_for_adaptive_sdm_gamma_body(
            prob.data,
            multiplicity.idx.data,
            len(multiplicity),
            multiplicity.data,
            cell_id.data,
            dt_left.data,
            dt,
            dt_range,
            is_first_in_pair.indicator.data,
            stats_n_substep.data,
            stats_dt_min.data,
        )

    @staticmethod
    # @numba.njit(**conf.JIT_FLAGS)  # note: as of Numba 0.51, np.dot() does not support ints
    def __cell_id_body(cell_id, cell_origin, strides):
        cell_id[:] = np.dot(strides, cell_origin)

    def cell_id(self, cell_id, cell_origin, strides):
        return self.__cell_id_body(cell_id.data, cell_origin.data, strides.data)

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def __collision_coalescence_body(
        *,
        multiplicity,
        idx,
        length,
        attributes,
        gamma,
        healthy,
        cell_id,
        coalescence_rate,
        is_first_in_pair,
    ):
        for i in numba.prange(  # pylint: disable=not-an-iterable,too-many-nested-blocks
            length // 2
        ):
            j, k, skip_pair = pair_indices(i, idx, is_first_in_pair, gamma)
            if skip_pair:
                continue
            coalesce(
                i, j, k, cell_id[j], multiplicity, gamma, attributes, coalescence_rate
            )
            flag_zero_multiplicity(j, k, multiplicity, healthy)

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
        self.__collision_coalescence_body(
            multiplicity=multiplicity.data,
            idx=idx.data,
            length=len(idx),
            attributes=attributes.data,
            gamma=gamma.data,
            healthy=healthy.data,
            cell_id=cell_id.data,
            coalescence_rate=coalescence_rate.data,
            is_first_in_pair=is_first_in_pair.indicator.data,
        )

    def collision_coalescence_breakup(
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
        # pylint: disable=too-many-locals
        self.__collision_coalescence_breakup_body(
            multiplicity=multiplicity.data,
            idx=idx.data,
            length=len(idx),
            attributes=attributes.data,
            gamma=gamma.data,
            rand=rand.data,
            Ec=Ec.data,
            Eb=Eb.data,
            fragment_mass=fragment_mass.data,
            healthy=healthy.data,
            cell_id=cell_id.data,
            coalescence_rate=coalescence_rate.data,
            breakup_rate=breakup_rate.data,
            breakup_rate_deficit=breakup_rate_deficit.data,
            is_first_in_pair=is_first_in_pair.indicator.data,
            max_multiplicity=max_multiplicity,
            warn_overflows=warn_overflows,
            particle_mass=particle_mass.data,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    # pylint: disable=too-many-arguments
    def __fragmentation_limiters(n_fragment, frag_volume, vmin, nfmax, x_plus_y):
        for i in numba.prange(len(frag_volume)):  # pylint: disable=not-an-iterable
            if x_plus_y[i] == 0.0:
                frag_volume[i] = 0.0
                n_fragment[i] = 1.0
            else:
                if np.isnan(frag_volume[i]) or frag_volume[i] == 0.0:
                    frag_volume[i] = x_plus_y[i]
                frag_volume[i] = min(frag_volume[i], x_plus_y[i])
                if nfmax is not None and x_plus_y[i] / frag_volume[i] > nfmax:
                    frag_volume[i] = x_plus_y[i] / nfmax
                elif frag_volume[i] < vmin:
                    frag_volume[i] = x_plus_y[i]
                n_fragment[i] = x_plus_y[i] / frag_volume[i]

    def fragmentation_limiters(self, *, n_fragment, frag_volume, vmin, nfmax, x_plus_y):
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            vmin=vmin,
            nfmax=nfmax,
            x_plus_y=x_plus_y.data,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    def __slams_fragmentation_body(n_fragment, frag_volume, x_plus_y, probs, rand):
        for i in numba.prange(len(n_fragment)):  # pylint: disable=not-an-iterable
            probs[i] = 0.0
            n_fragment[i] = 1
            for n in range(22):
                probs[i] += 0.91 * (n + 2) ** (-1.56)
                if rand[i] < probs[i]:
                    n_fragment[i] = n + 2
                    break
            frag_volume[i] = x_plus_y[i] / n_fragment[i]

    def slams_fragmentation(
        self, n_fragment, frag_volume, x_plus_y, probs, rand, vmin, nfmax
    ):  # pylint: disable=too-many-arguments
        self.__slams_fragmentation_body(
            n_fragment.data, frag_volume.data, x_plus_y.data, probs.data, rand.data
        )
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            vmin=vmin,
            nfmax=nfmax,
            x_plus_y=x_plus_y.data,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    # pylint: disable=too-many-arguments
    def __exp_fragmentation_body(*, scale, frag_volume, rand, tol=1e-5):
        """
        Exponential PDF
        """
        for i in numba.prange(len(frag_volume)):  # pylint: disable=not-an-iterable
            frag_volume[i] = -scale * np.log(max(1 - rand[i], tol))

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
        self.__exp_fragmentation_body(
            scale=scale,
            frag_volume=frag_volume.data,
            rand=rand.data,
            tol=tol,
        )
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
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
        self.__feingold1988_fragmentation_body(
            scale=scale,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            rand=rand.data,
            fragtol=fragtol,
        )

        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    def gauss_fragmentation(
        self, *, n_fragment, mu, sigma, frag_volume, x_plus_y, rand, vmin, nfmax
    ):
        self.__gauss_fragmentation_body(
            mu=mu,
            sigma=sigma,
            frag_volume=frag_volume.data,
            rand=rand.data,
        )
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
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
        self.__straub_fragmentation_body(
            CW=CW.data,
            gam=gam.data,
            ds=ds.data,
            frag_volume=frag_volume.data,
            v_max=v_max.data,
            rand=rand.data,
            Nr1=Nr1.data,
            Nr2=Nr2.data,
            Nr3=Nr3.data,
            Nr4=Nr4.data,
            Nrt=Nrt.data,
            d34=d34.data,
        )
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    def ll82_fragmentation(
        # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        n_fragment,
        CKE,
        W,
        W2,
        St,
        ds,
        dl,
        dcoal,
        frag_volume,
        x_plus_y,
        rand,
        vmin,
        nfmax,
        Rf,
        Rs,
        Rd,
        tol=1e-8,
    ):
        self.__ll82_fragmentation_body(
            CKE=CKE.data,
            W=W.data,
            W2=W2.data,
            St=St.data,
            ds=ds.data,
            dl=dl.data,
            dcoal=dcoal.data,
            frag_volume=frag_volume.data,
            rand=rand.data,
            Rf=Rf.data,
            Rs=Rs.data,
            Rd=Rd.data,
            tol=tol,
        )
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_volume=frag_volume.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    def ll82_coalescence_check(self, *, Ec, dl):
        self.__ll82_coalescence_check_body(
            Ec=Ec.data,
            dl=dl.data,
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments,too-many-locals
    def __compute_gamma_body(
        prob,
        rand,
        idx,
        length,
        multiplicity,
        cell_id,
        collision_rate_deficit,
        collision_rate,
        is_first_in_pair,
        out,
    ):
        """
        return in "out" array gamma (see: http://doi.org/10.1002/qj.441, section 5)
        formula:
        gamma = floor(prob) + 1 if rand <  prob - floor(prob)
              = floor(prob)     if rand >= prob - floor(prob)

        out may point to the same array as prob
        """
        for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
            out[i] = np.ceil(prob[i] - rand[i])
            j, k, skip_pair = pair_indices(i, idx, is_first_in_pair, out)
            if skip_pair:
                continue
            prop = multiplicity[j] // multiplicity[k]
            g = min(int(out[i]), prop)
            cid = cell_id[j]
            atomic_add(collision_rate, cid, g * multiplicity[k])
            atomic_add(collision_rate_deficit, cid, (int(out[i]) - g) * multiplicity[k])
            out[i] = g

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
        return self.__compute_gamma_body(
            prob.data,
            rand.data,
            multiplicity.idx.data,
            len(multiplicity),
            multiplicity.data,
            cell_id.data,
            collision_rate_deficit.data,
            collision_rate.data,
            is_first_in_pair.indicator.data,
            out.data,
        )

    @staticmethod
    def make_cell_caretaker(idx_shape, idx_dtype, cell_start_len, scheme="default"):
        class CellCaretaker:  # pylint: disable=too-few-public-methods
            def __init__(self, idx_shape, idx_dtype, cell_start_len, scheme):
                if scheme == "default":
                    if conf.JIT_FLAGS["parallel"]:
                        scheme = "counting_sort_parallel"
                    else:
                        scheme = "counting_sort"
                self.scheme = scheme
                if scheme in ("counting_sort", "counting_sort_parallel"):
                    self.tmp_idx = Storage.empty(idx_shape, idx_dtype)
                if scheme == "counting_sort_parallel":
                    self.cell_starts = Storage.empty(
                        (
                            numba.config.NUMBA_NUM_THREADS,  # pylint: disable=no-member
                            cell_start_len,
                        ),
                        dtype=int,
                    )

            def __call__(self, cell_id, cell_idx, cell_start, idx):
                length = len(idx)
                if self.scheme == "counting_sort":
                    CollisionsMethods._counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data,
                        idx.data,
                        cell_id.data,
                        cell_idx.data,
                        length,
                        cell_start.data,
                    )
                elif self.scheme == "counting_sort_parallel":
                    CollisionsMethods._parallel_counting_sort_by_cell_id_and_update_cell_start(
                        self.tmp_idx.data,
                        idx.data,
                        cell_id.data,
                        cell_idx.data,
                        length,
                        cell_start.data,
                        self.cell_starts.data,
                    )
                idx.data, self.tmp_idx.data = self.tmp_idx.data, idx.data

        return CellCaretaker(idx_shape, idx_dtype, cell_start_len, scheme)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
    # pylint: disable=too-many-arguments
    def __normalize_body(
        prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv
    ):
        n_cell = cell_start.shape[0] - 1
        for i in range(n_cell):
            sd_num = cell_start[i + 1] - cell_start[i]
            if sd_num < 2:
                norm_factor[i] = 0
            else:
                norm_factor[i] = (
                    timestep / dv * sd_num * (sd_num - 1) / 2 / (sd_num // 2)
                )
        for d in numba.prange(prob.shape[0]):  # pylint: disable=not-an-iterable
            prob[d] *= norm_factor[cell_idx[cell_id[d]]]

    # pylint: disable=too-many-arguments
    def normalize(self, prob, cell_id, cell_idx, cell_start, norm_factor, timestep, dv):
        return self.__normalize_body(
            prob.data,
            cell_id.data,
            cell_idx.data,
            cell_start.data,
            norm_factor.data,
            timestep,
            dv,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
    def remove_zero_n_or_flagged(multiplicity, idx, length) -> int:
        flag = len(idx)
        new_length = length
        i = 0
        while i < new_length:
            if idx[i] == flag or multiplicity[idx[i]] == 0:
                new_length -= 1
                idx[i] = idx[new_length]
                idx[new_length] = flag
            else:
                i += 1
        return new_length

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments
    def _counting_sort_by_cell_id_and_update_cell_start(
        new_idx, idx, cell_id, cell_idx, length, cell_start
    ):
        cell_end = cell_start
        # Warning: Assuming len(cell_end) == n_cell+1
        cell_end[:] = 0
        for i in range(length):
            cell_end[cell_idx[cell_id[idx[i]]]] += 1
        for i in range(1, len(cell_end)):
            cell_end[i] += cell_end[i - 1]
        for i in range(length - 1, -1, -1):
            cell_end[cell_idx[cell_id[idx[i]]]] -= 1
            new_idx[cell_end[cell_idx[cell_id[idx[i]]]]] = idx[i]

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    # pylint: disable=too-many-arguments
    def _parallel_counting_sort_by_cell_id_and_update_cell_start(
        new_idx, idx, cell_id, cell_idx, length, cell_start, cell_start_p
    ):
        cell_end_thread = cell_start_p
        # Warning: Assuming len(cell_end) == n_cell+1
        thread_num = cell_end_thread.shape[0]
        for t in numba.prange(thread_num):  # pylint: disable=not-an-iterable
            cell_end_thread[t, :] = 0
            for i in range(
                t * length // thread_num,
                (t + 1) * length // thread_num if t < thread_num - 1 else length,
            ):
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] += 1

        cell_start[:] = np.sum(cell_end_thread, axis=0)
        for i in range(1, len(cell_start)):
            cell_start[i] += cell_start[i - 1]

        tmp = cell_end_thread[0, :]
        tmp[:] = cell_end_thread[thread_num - 1, :]
        cell_end_thread[thread_num - 1, :] = cell_start[:]
        for t in range(thread_num - 2, -1, -1):
            cell_start[:] = cell_end_thread[t + 1, :] - tmp[:]
            tmp[:] = cell_end_thread[t, :]
            cell_end_thread[t, :] = cell_start[:]

        for t in numba.prange(thread_num):  # pylint: disable=not-an-iterable
            for i in range(
                (
                    (t + 1) * length // thread_num - 1
                    if t < thread_num - 1
                    else length - 1
                ),
                t * length // thread_num - 1,
                -1,
            ):
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] -= 1
                new_idx[cell_end_thread[t, cell_idx[cell_id[idx[i]]]]] = idx[i]

        cell_start[:] = cell_end_thread[0, :]

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
                        D = D1 / r**D2
                        E = E1 / r**E2
                        F = (F1 / r) ** Mf + F2
                        output[i // 2] = A + B * p + D / p**F + E / Gp
                        output[i // 2] = max(0, output[i // 2])

    def linear_collection_efficiency(
        self, *, params, output, radii, is_first_in_pair, unit
    ):
        return self.linear_collection_efficiency_body(
            params,
            output.data,
            radii.data,
            is_first_in_pair.indicator.data,
            radii.idx.data,
            len(is_first_in_pair),
            unit,
        )
