"""
Koehler-curve equilibrium in unsaturated conditions
"""

import numba
import numpy as np

from ..backends.impl_numba.conf import JIT_FLAGS
from ..backends.impl_numba.toms748 import toms748_solve
from ..backends.impl_numba.warnings import warn

default_rtol = 1e-5
default_max_iters = 64


def equilibrate_dry_radii(
    *,
    r_wet: np.ndarray,
    environment,
    kappa: np.ndarray,
    f_org: np.ndarray = None,
    cell_id: np.ndarray = None,
    rtol=default_rtol,
    max_iters=default_max_iters,
):
    if cell_id is None:
        cell_id = np.zeros_like(r_wet, dtype=int)
    if f_org is None:
        f_org = np.zeros_like(r_wet, dtype=float)

    formulae = environment.particulator.formulae
    jit_flags = {**JIT_FLAGS, **{"fastmath": formulae.fastmath, "cache": False}}

    const = environment.particulator.formulae.constants
    T = environment["T"].to_ndarray()
    RH = environment["RH"].to_ndarray()

    RH_eq = formulae.hygroscopicity.RH_eq
    sigma = formulae.surface_tension.sigma
    phys_volume = formulae.trivia.volume
    within_tolerance = formulae.trivia.within_tolerance

    @numba.njit(**{**jit_flags, "parallel": False})
    def minfun_dry(r_dry, T, RH, kp, r, f_org):  # pylint: disable=too-many-arguments
        rd3 = r_dry**3
        sgm = sigma(T, phys_volume(radius=r), const.PI_4_3 * rd3, f_org)
        return RH - RH_eq(r, T, kp, rd3, sgm)

    @numba.njit(**jit_flags)
    def r_dry_init_impl(  # pylint: disable=too-many-arguments,too-many-locals
        r_wet: np.ndarray,
        iters,
        T,
        RH,
        cell_id: np.ndarray,
        kappa,
        rtol,
        RH_range=(0, 1),
    ):
        r_dry = np.empty_like(r_wet)
        for i in numba.prange(len(r_dry)):  # pylint: disable=not-an-iterable
            cid = cell_id[i]

            # root-finding initial guess
            a = 0
            b = r_wet[i]

            # minimisation
            args = (
                T[cid],
                np.maximum(RH_range[0], np.minimum(RH_range[1], RH[cid])),
                kappa[i],
                r_wet[i],
                f_org[i],
            )

            fa = minfun_dry(a, *args)
            fb = minfun_dry(b, *args)

            r_dry[i], iters[i] = toms748_solve(
                minfun_dry,
                args,
                a,
                b,
                fa,
                fb,
                rtol=rtol,
                max_iter=max_iters,
                within_tolerance=within_tolerance,
            )
            if iters[i] == -1:
                warn(
                    msg="failed to find dry radius for particle",
                    file=__file__,
                    context=(
                        "i",
                        i,
                        "r_w",
                        r_wet[i],
                        "T",
                        T[cid],
                        "RH",
                        RH[cid],
                        "f_org",
                        f_org[i],
                        "kappa",
                        kappa[i],
                    ),
                )
        return r_dry

    iters = np.empty_like(r_wet, dtype=int)
    r_dry = r_dry_init_impl(
        r_wet=r_wet, iters=iters, T=T, RH=RH, cell_id=cell_id, kappa=kappa, rtol=rtol
    )
    assert (iters != max_iters).all() and (iters != -1).all()
    return r_dry


def equilibrate_wet_radii(
    *,
    r_dry: np.ndarray,
    environment,
    kappa_times_dry_volume: np.ndarray,
    f_org: np.ndarray = None,
    cell_id: np.ndarray = None,
    rtol=default_rtol,
    max_iters=default_max_iters,
):
    # pylint: disable=too-many-locals
    if cell_id is None:
        cell_id = np.zeros_like(r_dry, dtype=int)
    if f_org is None:
        f_org = np.zeros_like(r_dry, dtype=float)

    const = environment.particulator.formulae.constants
    T = environment["T"].to_ndarray()
    RH = environment["RH"].to_ndarray()

    formulae = environment.particulator.formulae
    r_cr = formulae.hygroscopicity.r_cr
    RH_eq = formulae.hygroscopicity.RH_eq
    sigma = formulae.surface_tension.sigma
    phys_volume = formulae.trivia.volume
    within_tolerance = formulae.trivia.within_tolerance

    kappa = kappa_times_dry_volume / phys_volume(radius=r_dry)

    jit_flags = {**JIT_FLAGS, **{"fastmath": formulae.fastmath, "cache": False}}

    @numba.njit(**{**jit_flags, "parallel": False})
    def minfun_wet(r, T, RH, kp, rd3, f_org):  # pylint: disable=too-many-arguments
        sgm = sigma(T, phys_volume(radius=r), const.PI_4_3 * rd3, f_org)
        return RH - RH_eq(r, T, kp, rd3, sgm)

    @numba.njit(**jit_flags)
    def r_wet_init_impl(  # pylint: disable=too-many-arguments,too-many-locals
        r_dry: np.ndarray,
        iters,
        T,
        RH,
        cell_id: np.ndarray,
        kappa,
        rtol,
        RH_range=(0, 1),
    ):
        r_wet = np.empty_like(r_dry)
        for i in numba.prange(len(r_dry)):  # pylint: disable=not-an-iterable
            cid = cell_id[i]

            # root-finding initial guess
            a = r_dry[i]
            b = r_cr(kappa[i], r_dry[i] ** 3, T[cid], const.sgm_w)

            if not a < b:
                r_wet[i] = r_dry[i]
                iters[i] = 0
                continue

            # minimisation
            args = (
                T[cid],
                np.maximum(RH_range[0], np.minimum(RH_range[1], RH[cid])),
                kappa[i],
                r_dry[i] ** 3,
                f_org[i],
            )

            fa = minfun_wet(a, *args)
            if fa < 0:
                r_wet[i] = r_dry[i]
                iters[i] = 0
                continue
            fb = minfun_wet(b, *args)

            r_wet[i], iters[i] = toms748_solve(
                minfun_wet,
                args,
                a,
                b,
                fa,
                fb,
                rtol=rtol,
                max_iter=max_iters,
                within_tolerance=within_tolerance,
            )
            if iters[i] == -1:
                warn(
                    msg="failed to find wet radius for particle",
                    file=__file__,
                    context=(
                        "i",
                        i,
                        "r_d",
                        r_dry[i],
                        "T",
                        T[cid],
                        "RH",
                        RH[cid],
                        "f_org",
                        f_org[i],
                        "kappa",
                        kappa[i],
                    ),
                )
        return r_wet

    iters = np.empty_like(r_dry, dtype=int)
    r_wet = r_wet_init_impl(
        r_dry=r_dry, iters=iters, T=T, RH=RH, cell_id=cell_id, kappa=kappa, rtol=rtol
    )
    assert (iters != max_iters).all() and (iters != -1).all()
    return r_wet
