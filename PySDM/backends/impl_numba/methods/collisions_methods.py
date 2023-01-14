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
def breakup_fun0(
    gamma, j, k, multiplicity, volume, nfi, fragment_size_i, max_multiplicity
):  # pylint: disable=too-many-arguments
    overflow_flag = False
    gamma_j_k = 0
    take_from_j_test = multiplicity[k]
    take_from_j = 0
    new_mult_k_test = 0
    new_mult_k = multiplicity[k]
    for m in range(int(gamma)):
        take_from_j_test += new_mult_k_test
        new_mult_k_test *= volume[j] / fragment_size_i
        new_mult_k_test += nfi * multiplicity[k]

        # check for overflow of multiplicity
        if new_mult_k_test > max_multiplicity:
            overflow_flag = True
            break

        # check for new_n > 0
        if take_from_j_test > multiplicity[j]:
            break

        take_from_j = take_from_j_test
        new_mult_k = new_mult_k_test
        gamma_j_k = m + 1
    return take_from_j, new_mult_k, gamma_j_k, overflow_flag


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def breakup_fun1(
    j, k, attributes, multiplicity, take_from_j, new_mult_k
):  # pylint: disable=too-many-arguments
    for a in range(len(attributes)):
        attributes[a, k] *= multiplicity[k]
        attributes[a, k] += take_from_j * attributes[a, j]
        attributes[a, k] /= new_mult_k

    if multiplicity[j] > take_from_j:
        nj = multiplicity[j] - take_from_j
        nk = new_mult_k
    else:
        nj = new_mult_k / 2
        nk = nj
    return nj, nk


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def breakup_fun2(
    j, k, nj, nk, attributes, multiplicity, take_from_j
):  # pylint: disable=too-many-arguments
    if multiplicity[j] <= take_from_j:
        for a in range(len(attributes)):
            attributes[a, j] = attributes[a, k]

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
    n_fragment,
    fragment_size,
    max_multiplicity,
    breakup_rate,
    breakup_rate_deficit,
    warn_overflows,
    volume,
):
    take_from_j, new_mult_k, gamma_j_k, overflow_flag = breakup_fun0(
        gamma[i],
        j,
        k,
        multiplicity,
        volume,
        n_fragment[i],
        fragment_size[i],
        max_multiplicity,
    )
    gamma_deficit = gamma[i] - gamma_j_k

    nj, nk = breakup_fun1(j, k, attributes, multiplicity, take_from_j, new_mult_k)

    if multiplicity[j] <= take_from_j and round(nj) == 0:
        atomic_add(breakup_rate_deficit, cid, gamma[i] * multiplicity[k])
        return

    atomic_add(breakup_rate, cid, gamma_j_k * multiplicity[k])
    atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])

    breakup_fun2(j, k, nj, nk, attributes, multiplicity, take_from_j)

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
    n_fragment,
    fragment_size,
    max_multiplicity,
    breakup_rate,
    breakup_rate_deficit,
    warn_overflows,
    volume,
):  # pylint: disable=too-many-arguments,unused-argument,too-many-locals
    gamma_deficit = gamma[i]
    overflow_flag = False
    while gamma_deficit > 0:
        if multiplicity[k] == multiplicity[j]:
            take_from_j = multiplicity[j]
            new_mult_k = (volume[j] + volume[k]) / fragment_size[i] * multiplicity[k]

            # check for overflow
            if new_mult_k > max_multiplicity:
                atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])
                overflow_flag = True
                break
            gamma_j_k = gamma_deficit

        else:
            if multiplicity[k] > multiplicity[j]:
                j, k = k, j
            take_from_j, new_mult_k, gamma_j_k, overflow_flag = breakup_fun0(
                gamma_deficit,
                j,
                k,
                multiplicity,
                volume,
                (volume[j] + volume[k]) / fragment_size[i],
                fragment_size[i],
                max_multiplicity,
            )

        nj, nk = breakup_fun1(j, k, attributes, multiplicity, take_from_j, new_mult_k)

        if multiplicity[j] <= take_from_j and round(nj) == 0:
            atomic_add(breakup_rate_deficit, cid, gamma[i] * multiplicity[k])
            return

        atomic_add(breakup_rate, cid, gamma_j_k * multiplicity[k])
        gamma_deficit -= gamma_j_k
        breakup_fun2(j, k, nj, nk, attributes, multiplicity, take_from_j)

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


class CollisionsMethods(BackendMethods):
    def __init__(self):
        BackendMethods.__init__(self)

        _break_up = break_up_while if self.formulae.handle_all_breakups else break_up

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
            n_fragment,
            fragment_size,
            healthy,
            cell_id,
            coalescence_rate,
            breakup_rate,
            breakup_rate_deficit,
            is_first_in_pair,
            max_multiplicity,
            warn_overflows,
            volume,
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
                        n_fragment,
                        fragment_size,
                        max_multiplicity,
                        breakup_rate,
                        breakup_rate_deficit,
                        warn_overflows,
                        volume,
                    )
                flag_zero_multiplicity(j, k, multiplicity, healthy)

        self.__collision_coalescence_breakup_body = __collision_coalescence_breakup_body

        if self.formulae.fragmentation_function.__name__ == "Straub2010Nf":
            straub_p1 = self.formulae.fragmentation_function.p1
            straub_p2 = self.formulae.fragmentation_function.p2
            straub_p3 = self.formulae.fragmentation_function.p3
            straub_p4 = self.formulae.fragmentation_function.p4
            straub_sigma1 = self.formulae.fragmentation_function.sigma1

            @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
            def __straub_fragmentation_body(
                *, CW, gam, ds, v_max, frag_size, rand, Nr1, Nr2, Nr3, Nr4, Nrt
            ):
                for i in numba.prange(  # pylint: disable=not-an-iterable
                    len(frag_size)
                ):
                    straub_Nr(i, Nr1, Nr2, Nr3, Nr4, Nrt, CW, gam)
                    if rand[i] < Nr1[i] / Nrt[i]:
                        frag_size[i] = straub_p1(
                            rand[i] * Nrt[i] / Nr1[i], straub_sigma1(CW[i])
                        )
                    elif rand[i] < (Nr2[i] + Nr1[i]) / Nrt[i]:
                        frag_size[i] = straub_p2(
                            CW[i], (rand[i] * Nrt[i] - Nr1[i]) / (Nr2[i] - Nr1[i])
                        )
                    elif rand[i] < (Nr3[i] + Nr2[i] + Nr1[i]) / Nrt[i]:
                        frag_size[i] = straub_p3(
                            CW[i],
                            ds[i],
                            (rand[i] * Nrt[i] - Nr2[i]) / (Nr3[i] - Nr2[i]),
                        )
                    else:
                        frag_size[i] = straub_p4(
                            CW[i], ds[i], v_max[i], Nr1[i], Nr2[i], Nr3[i]
                        )

            self.__straub_fragmentation_body = __straub_fragmentation_body
        elif self.formulae.fragmentation_function.__name__ == "Gaussian":
            gaussian_frag_size = self.formulae.fragmentation_function.frag_size

            @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
            def __gauss_fragmentation_body(
                *, mu, sigma, frag_size, rand
            ):  # pylint: disable=too-many-arguments
                for i in numba.prange(  # pylint: disable=not-an-iterable
                    len(frag_size)
                ):
                    frag_size[i] = gaussian_frag_size(mu, sigma, rand[i])

            self.__gauss_fragmentation_body = __gauss_fragmentation_body
        elif self.formulae.fragmentation_function.__name__ == "Feingold1988Frag":
            feingold1988_frag_size = self.formulae.fragmentation_function.frag_size

            @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
            # pylint: disable=too-many-arguments
            def __feingold1988_fragmentation_body(
                *, scale, frag_size, x_plus_y, rand, fragtol
            ):
                for i in numba.prange(  # pylint: disable=not-an-iterable
                    len(frag_size)
                ):
                    frag_size[i] = feingold1988_frag_size(
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
        n,
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
            n.idx.data,
            len(n),
            n.data,
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
        n_fragment,
        fragment_size,
        healthy,
        cell_id,
        coalescence_rate,
        breakup_rate,
        breakup_rate_deficit,
        is_first_in_pair,
        warn_overflows,
        volume,
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
            n_fragment=n_fragment.data,
            fragment_size=fragment_size.data,
            healthy=healthy.data,
            cell_id=cell_id.data,
            coalescence_rate=coalescence_rate.data,
            breakup_rate=breakup_rate.data,
            breakup_rate_deficit=breakup_rate_deficit.data,
            is_first_in_pair=is_first_in_pair.indicator.data,
            max_multiplicity=max_multiplicity,
            warn_overflows=warn_overflows,
            volume=volume.data,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    # pylint: disable=too-many-arguments
    def __fragmentation_limiters(n_fragment, frag_size, v_max, vmin, nfmax, x_plus_y):
        for i in numba.prange(len(frag_size)):  # pylint: disable=not-an-iterable
            frag_size[i] = min(frag_size[i], v_max[i])
            frag_size[i] = max(frag_size[i], vmin)
            if nfmax is not None:
                if x_plus_y[i] / frag_size[i] > nfmax:
                    frag_size[i] = x_plus_y[i] / nfmax
            if frag_size[i] == 0.0:
                frag_size[i] = x_plus_y[i]
            n_fragment[i] = x_plus_y[i] / frag_size[i]

    def fragmentation_limiters(
        self, *, n_fragment, frag_size, v_max, vmin, nfmax, x_plus_y
    ):
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_size=frag_size.data,
            v_max=v_max.data,
            vmin=vmin,
            nfmax=nfmax,
            x_plus_y=x_plus_y.data,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    def __slams_fragmentation_body(n_fragment, frag_size, x_plus_y, probs, rand):
        for i in numba.prange(len(n_fragment)):  # pylint: disable=not-an-iterable
            probs[i] = 0.0
            n_fragment[i] = 1
            for n in range(22):
                probs[i] += 0.91 * (n + 2) ** (-1.56)
                if rand[i] < probs[i]:
                    n_fragment[i] = n + 2
                    break
            frag_size[i] = x_plus_y[i] / n_fragment[i]

    def slams_fragmentation(
        self, n_fragment, frag_size, v_max, x_plus_y, probs, rand, vmin, nfmax
    ):  # pylint: disable=too-many-arguments
        self.__slams_fragmentation_body(
            n_fragment.data, frag_size.data, x_plus_y.data, probs.data, rand.data
        )
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_size=frag_size.data,
            v_max=v_max.data,
            vmin=vmin,
            nfmax=nfmax,
            x_plus_y=x_plus_y.data,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    # pylint: disable=too-many-arguments
    def __exp_fragmentation_body(*, scale, frag_size, rand, tol=1e-5):
        """
        Exponential PDF
        """
        for i in numba.prange(len(frag_size)):  # pylint: disable=not-an-iterable
            frag_size[i] = -scale * np.log(max(1 - rand[i], tol))

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
        self.__exp_fragmentation_body(
            scale=scale,
            frag_size=frag_size.data,
            rand=rand.data,
            tol=tol,
        )
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_size=frag_size.data,
            v_max=v_max.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
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
        self.__feingold1988_fragmentation_body(
            scale=scale,
            frag_size=frag_size.data,
            x_plus_y=x_plus_y.data,
            rand=rand.data,
            fragtol=fragtol,
        )

        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_size=frag_size.data,
            v_max=v_max.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
        )

    def gauss_fragmentation(
        self, *, n_fragment, mu, sigma, frag_size, v_max, x_plus_y, rand, vmin, nfmax
    ):
        self.__gauss_fragmentation_body(
            mu=mu,
            sigma=sigma,
            frag_size=frag_size.data,
            rand=rand.data,
        )
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_size=frag_size.data,
            v_max=v_max.data,
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
        self.__straub_fragmentation_body(
            CW=CW.data,
            gam=gam.data,
            ds=ds.data,
            frag_size=frag_size.data,
            v_max=v_max.data,
            rand=rand.data,
            Nr1=Nr1.data,
            Nr2=Nr2.data,
            Nr3=Nr3.data,
            Nr4=Nr4.data,
            Nrt=Nrt.data,
        )
        self.__fragmentation_limiters(
            n_fragment=n_fragment.data,
            frag_size=frag_size.data,
            v_max=v_max.data,
            x_plus_y=x_plus_y.data,
            vmin=vmin,
            nfmax=nfmax,
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
    def make_cell_caretaker(idx, cell_start, scheme="default"):
        class CellCaretaker:  # pylint: disable=too-few-public-methods
            def __init__(self, idx, cell_start, scheme):
                if scheme == "default":
                    if conf.JIT_FLAGS["parallel"]:
                        scheme = "counting_sort_parallel"
                    else:
                        scheme = "counting_sort"
                self.scheme = scheme
                if scheme in ("counting_sort", "counting_sort_parallel"):
                    self.tmp_idx = Storage.empty(idx.shape, idx.dtype)
                if scheme == "counting_sort_parallel":
                    self.cell_starts = Storage.empty(
                        (
                            numba.config.NUMBA_NUM_THREADS,  # pylint: disable=no-member
                            len(cell_start),
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

        return CellCaretaker(idx, cell_start, scheme)

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
                (t + 1) * length // thread_num - 1
                if t < thread_num - 1
                else length - 1,
                t * length // thread_num - 1,
                -1,
            ):
                cell_end_thread[t, cell_idx[cell_id[idx[i]]]] -= 1
                new_idx[cell_end_thread[t, cell_idx[cell_id[idx[i]]]]] = idx[i]

        cell_start[:] = cell_end_thread[0, :]
