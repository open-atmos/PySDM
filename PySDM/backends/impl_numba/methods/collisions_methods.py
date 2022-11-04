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
from PySDM.physics.constants import PI, PI_4_3, si, sqrt_pi, sqrt_two

CM = si.cm


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def pair_indices(i, idx, is_first_in_pair):
    """given permutation array `idx` and `is_first_in_pair` flag array,
    returns indices `j` and `k` of droplets within pair `i`
    such that `j` points to the droplet with higher (or equal) multiplicity
    """
    offset = 1 - is_first_in_pair[2 * i]
    j = idx[2 * i + offset]
    k = idx[2 * i + 1 + offset]
    return j, k


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
        for a in range(0, len(attributes)):
            attributes[a, k] += gamma[i] * attributes[a, j]
    else:  # new_n == 0
        multiplicity[j] = multiplicity[k] // 2
        multiplicity[k] = multiplicity[k] - multiplicity[j]
        for a in range(0, len(attributes)):
            attributes[a, j] = gamma[i] * attributes[a, j] + attributes[a, k]
            attributes[a, k] = attributes[a, j]


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def break_up(  # pylint: disable=too-many-arguments,unused-argument,too-many-locals
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
):  # pylint: disable=too-many-branches
    overflow_flag = False
    take_from_j_test = multiplicity[k]
    new_mult_k_test = 0
    new_mult_k = multiplicity[k]
    take_from_j = 0
    gamma_tmp = 0
    gamma_deficit = gamma[i]
    for m in range(int(gamma[i])):
        take_from_j_test = new_mult_k_test + take_from_j_test
        new_mult_k_test = (
            new_mult_k_test * (volume[j] / fragment_size[i])
            + n_fragment[i] * multiplicity[k]
        )
        # check for overflow of multiplicity
        if new_mult_k_test > max_multiplicity:
            overflow_flag = True
            break
        # check for new_n > 0
        if take_from_j_test > multiplicity[j]:
            break

        # all tests passed
        take_from_j = take_from_j_test
        new_mult_k = new_mult_k_test
        gamma_tmp = m + 1
        gamma_deficit = gamma[i] - gamma_tmp
    # 2. Compute the new multiplicities and particle sizes, with rounding
    for a in range(0, len(attributes)):
        attributes[a, k] *= multiplicity[k]
        attributes[a, k] += take_from_j * attributes[a, j]
        attributes[a, k] /= new_mult_k
    if multiplicity[j] > take_from_j:
        nj = multiplicity[j] - take_from_j
        nk = new_mult_k
    else:
        nj = new_mult_k / 2
        if round(nj) == 0:
            atomic_add(breakup_rate_deficit, cid, gamma[i] * multiplicity[k])
            return
        nk = nj
        for a in range(0, len(attributes)):
            attributes[a, j] = attributes[a, k]
    # add up the product
    atomic_add(breakup_rate, cid, gamma_tmp * multiplicity[k])
    atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])
    # perform rounding as necessary
    multiplicity[j] = max(round(nj), 1)
    multiplicity[k] = max(round(nk), 1)
    factor_j = nj / multiplicity[j]
    factor_k = nk / multiplicity[k]
    for a in range(0, len(attributes)):
        attributes[a, k] *= factor_k
        attributes[a, j] *= factor_j

    if overflow_flag:
        if warn_overflows:
            warn("overflow", __file__)


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def break_up_while(  # pylint: disable=too-many-arguments,unused-argument,too-many-statements,too-many-locals
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
):  # pylint: disable=too-many-branches
    gamma_tmp = 0
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
            gamma_tmp = gamma_deficit

        else:
            # reorder droplets if necessary
            if multiplicity[k] > multiplicity[j]:
                j, k = k, j
            take_from_j_test = multiplicity[k]
            take_from_j = 0
            new_mult_k_test = 0
            new_mult_k = multiplicity[k]
            for m in range(int(gamma_deficit)):
                take_from_j_test = new_mult_k_test + take_from_j_test
                nfi = (volume[j] + volume[k]) / fragment_size[i]
                new_mult_k_test = (
                    new_mult_k_test * (volume[j] / fragment_size[i])
                    + nfi * multiplicity[k]
                )
                # check for overflow of multiplicity
                if new_mult_k_test > max_multiplicity:
                    overflow_flag = True
                    break
                # check for new_n > 0
                if take_from_j_test > multiplicity[j]:
                    break

                # all tests passed
                take_from_j = take_from_j_test
                new_mult_k = new_mult_k_test
                gamma_tmp = m + 1
        # Compute the new multiplicities and particle sizes, with rounding
        for a in range(0, len(attributes)):
            attributes[a, k] *= multiplicity[k]
            attributes[a, k] += take_from_j * attributes[a, j]
            attributes[a, k] /= new_mult_k
        if multiplicity[j] > take_from_j:
            nj = multiplicity[j] - take_from_j
            nk = new_mult_k
        else:
            nj = new_mult_k / 2
            if round(nj) == 0:
                atomic_add(breakup_rate_deficit, cid, gamma_tmp * multiplicity[k])
                return
            nk = nj
            for a in range(0, len(attributes)):
                attributes[a, j] = attributes[a, k]

        atomic_add(breakup_rate, cid, gamma_tmp * multiplicity[k])
        # perform rounding as necessary
        multiplicity[j] = max(round(nj), 1)
        multiplicity[k] = max(round(nk), 1)
        factor_j = nj / multiplicity[j]
        factor_k = nk / multiplicity[k]
        for a in range(0, len(attributes)):
            attributes[a, k] *= factor_k
            attributes[a, j] *= factor_j
        gamma_deficit -= gamma_tmp

    atomic_add(breakup_rate_deficit, cid, gamma_deficit * multiplicity[k])

    if overflow_flag:
        if warn_overflows:
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
def straub_p1(  # pylint: disable=too-many-arguments,unused-argument
    i,
    CW,
    frag_size,
    rand,
):
    E_D1 = 0.04 * CM
    delD1 = 0.0125 * CW[i] ** (1 / 2)
    var_1 = delD1**2 / 12
    sigma1 = np.sqrt(np.log(var_1 / E_D1**2 + 1))
    mu1 = np.log(E_D1) - sigma1**2 / 2
    X = rand[i]

    frag_size[i] = np.exp(
        mu1
        - sigma1 / sqrt_two / sqrt_pi / np.log(2) * np.log((1 / 2 + X) / (3 / 2 - X))
    )
    frag_size[i] = PI / 6 * frag_size[i] ** 3


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def straub_p2(  # pylint: disable=too-many-arguments,unused-argument
    i,
    CW,
    frag_size,
    rand,
):
    mu2 = 0.095 * CM
    delD2 = 0.007 * (CW[i] - 21.0)
    sigma2 = delD2**2 / 12
    X = rand[i]

    frag_size[i] = mu2 - sigma2 / sqrt_two / sqrt_pi / np.log(2) * np.log(
        (1 / 2 + X) / (3 / 2 - X)
    )
    frag_size[i] = PI / 6 * frag_size[i] ** 3


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def straub_p3(  # pylint: disable=too-many-arguments,unused-argument
    i,
    CW,
    ds,
    frag_size,
    rand,
):
    mu3 = 0.9 * ds[i]
    delD3 = 0.01 * (0.76 * CW[i] ** (1 / 2) + 1.0)
    sigma3 = delD3**2 / 12
    X = rand[i]

    frag_size[i] = mu3 - sigma3 / sqrt_two / sqrt_pi / np.log(2) * np.log(
        (1 / 2 + X) / (3 / 2 - X)
    )
    frag_size[i] = PI / 6 * frag_size[i] ** 3


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def straub_p4(  # pylint: disable=too-many-arguments,unused-argument,too-many-locals
    i, CW, ds, v_max, frag_size, Nr1, Nr2, Nr3
):
    E_D1 = 0.04 * CM
    delD1 = 0.0125 * CW[i] ** (1 / 2)
    var_1 = delD1**2 / 12
    sigma1 = np.sqrt(np.log(var_1 / E_D1**2 + 1))
    mu1 = np.log(E_D1) - sigma1**2 / 2
    mu2 = 0.095 * CM
    delD2 = 0.007 * (CW[i] - 21.0)
    sigma2 = delD2**2 / 12
    mu3 = 0.9 * ds[i]
    delD3 = 0.01 * (0.76 * CW[i] ** (1 / 2) + 1.0)
    sigma3 = delD3**2 / 12

    M31 = Nr1[i] * np.exp(3 * mu1 + 9 * sigma1**2 / 2)
    M32 = Nr2[i] * (mu2**3 + 3 * mu2 * sigma2**2)
    M33 = Nr3[i] * (mu3**3 + 3 * mu3 * sigma3**2)

    M34 = v_max[i] / PI_4_3 * 8 + ds[i] ** 3 - M31 - M32 - M33
    frag_size[i] = PI / 6 * M34


class CollisionsMethods(BackendMethods):
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
    def __adaptive_sdm_gamma_body(
        gamma,
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
            if gamma[i] == 0:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)
            prop = multiplicity[j] // multiplicity[k]
            dt_optimal = dt * prop / gamma[i]
            cid = cell_id[j]
            dt_optimal = max(dt_optimal, dt_range[0])
            dt_todo[cid] = min(dt_todo[cid], dt_optimal)
            stats_dt_min[cid] = min(stats_dt_min[cid], dt_optimal)
        for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
            if gamma[i] == 0:
                continue
            j, _ = pair_indices(i, idx, is_first_in_pair)
            gamma[i] *= dt_todo[cell_id[j]] / dt
        for cid in numba.prange(len(dt_todo)):  # pylint: disable=not-an-iterable
            dt_left[cid] -= dt_todo[cid]
            if dt_todo[cid] > 0:
                stats_n_substep[cid] += 1

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
        return self.__adaptive_sdm_gamma_body(
            gamma.data,
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
            if gamma[i] == 0:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)
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

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
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
        handle_all_breakups,
    ):
        # pylint: disable=not-an-iterable,too-many-nested-blocks,too-many-locals
        for i in numba.prange(length // 2):
            if gamma[i] == 0:
                continue
            bouncing = rand[i] - (Ec[i] + (1 - Ec[i]) * (Eb[i])) > 0
            if bouncing:
                continue
            j, k = pair_indices(i, idx, is_first_in_pair)

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
            elif handle_all_breakups:
                break_up_while(
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
            else:
                break_up(
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
        handle_all_breakups,
    ):
        # pylint: disable=too-many-locals
        max_multiplicity = np.iinfo(multiplicity.data.dtype).max // 2e5
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
            handle_all_breakups=handle_all_breakups,
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
                n_fragment[i] = 1.0
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

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS})
    # pylint: disable=too-many-arguments
    def __feingold1988_fragmentation_body(*, scale, frag_size, x_plus_y, rand, fragtol):
        """
        Scaled exponential PDF
        """
        for i in numba.prange(len(frag_size)):  # pylint: disable=not-an-iterable
            log_arg = max(1 - rand[i] * scale / x_plus_y[i], fragtol)
            frag_size[i] = -scale * np.log(log_arg)

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

    @staticmethod
    # pylint: disable=too-many-arguments
    @numba.njit(**{**conf.JIT_FLAGS})
    def __gauss_fragmentation_body(*, mu, sigma, frag_size, rand):
        """
        Gaussian PDF
        CDF = 1/2(1 + erf(x/sqrt(2)));
        approximate as erf(x) ~ tanh(ax) with a = sqrt(pi)log(2) as in Vedder 1987
        """
        for i in numba.prange(len(frag_size)):  # pylint: disable=not-an-iterable
            frag_size[i] = mu - sigma / sqrt_two / sqrt_pi / np.log(2) * np.log(
                (1 / 2 + rand[i]) / (3 / 2 - rand[i])
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

    @staticmethod
    # pylint: disable=too-many-arguments
    @numba.njit(**(conf.JIT_FLAGS))
    def __straub_fragmentation_body(
        *, CW, gam, ds, v_max, frag_size, rand, Nr1, Nr2, Nr3, Nr4, Nrt
    ):
        for i in numba.prange(len(frag_size)):  # pylint: disable=not-an-iterable
            straub_Nr(i, Nr1, Nr2, Nr3, Nr4, Nrt, CW, gam)
            if rand[i] < Nr1[i] / Nrt[i]:
                rand[i] = rand[i] * Nrt[i] / Nr1[i]
                straub_p1(i, CW, frag_size, rand)
            elif rand[i] < (Nr2[i] + Nr1[i]) / Nrt[i]:
                rand[i] = (rand[i] * Nrt[i] - Nr1[i]) / (Nr2[i] - Nr1[i])
                straub_p2(i, CW, frag_size, rand)
            elif rand[i] < (Nr3[i] + Nr2[i] + Nr1[i]) / Nrt[i]:
                rand[i] = (rand[i] * Nrt[i] - Nr2[i]) / (Nr3[i] - Nr2[i])
                straub_p3(i, CW, ds, frag_size, rand)
            else:
                straub_p4(i, CW, ds, v_max, frag_size, Nr1, Nr2, Nr3)

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
        gamma,
        rand,
        idx,
        length,
        multiplicity,
        cell_id,
        collision_rate_deficit,
        collision_rate,
        is_first_in_pair,
    ):
        """
        return in "gamma" array gamma (see: http://doi.org/10.1002/qj.441, section 5)
        formula:
        gamma = floor(prob) + 1 if rand <  prob - floor(prob)
              = floor(prob)     if rand >= prob - floor(prob)
        """
        for i in numba.prange(length // 2):  # pylint: disable=not-an-iterable
            gamma[i] = np.ceil(gamma[i] - rand[i])

            no_collision = gamma[i] == 0
            if no_collision:
                continue

            j, k = pair_indices(i, idx, is_first_in_pair)
            prop = multiplicity[j] // multiplicity[k]
            g = min(int(gamma[i]), prop)
            cid = cell_id[j]
            atomic_add(collision_rate, cid, g * multiplicity[k])
            atomic_add(
                collision_rate_deficit, cid, (int(gamma[i]) - g) * multiplicity[k]
            )
            gamma[i] = g

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
        return self.__compute_gamma_body(
            gamma.data,
            rand.data,
            multiplicity.idx.data,
            len(multiplicity),
            multiplicity.data,
            cell_id.data,
            collision_rate_deficit.data,
            collision_rate.data,
            is_first_in_pair.indicator.data,
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
