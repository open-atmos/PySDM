"""
Koehler-curve equilibrium in unsaturated conditions
"""
from ..backends.numba.toms748 import toms748_solve
from ..physics import constants as const
from ..backends.numba.conf import JIT_FLAGS
from ..backends.numba.impl.warnings import warn
import numba
import numpy as np

default_rtol = 1e-5
default_max_iters = 64


def r_wet_init(r_dry: np.ndarray, environment,
               kappa_times_dry_volume: np.ndarray,
               f_org: np.ndarray = None,
               cell_id: np.ndarray = None,
               rtol=default_rtol, max_iters=default_max_iters):
    if cell_id is None:
        cell_id = np.zeros_like(r_dry, dtype=int)
    if f_org is None:
        f_org = np.zeros_like(r_dry, dtype=float)

    T = environment["T"].to_ndarray()
    RH = environment["RH"].to_ndarray()

    formulae = environment.particulator.formulae
    r_cr = formulae.hygroscopicity.r_cr
    RH_eq = formulae.hygroscopicity.RH_eq
    sigma = formulae.surface_tension.sigma
    phys_volume = formulae.trivia.volume
    within_tolerance = formulae.trivia.within_tolerance

    kappa = kappa_times_dry_volume / phys_volume(radius=r_dry)

    jit_flags = {**JIT_FLAGS, **{'fastmath': formulae.fastmath, 'cache': False}}

    @numba.njit(**{**jit_flags, 'parallel': False})
    def minfun(r, T, RH, kp, rd3, f_org):
        sgm = sigma(T, v_wet=phys_volume(radius=r), v_dry=const.pi_4_3 * rd3, f_org=f_org)
        return RH - RH_eq(r, T, kp, rd3, sgm)

    @numba.njit(**jit_flags)
    def r_wet_init_impl(r_dry: np.ndarray, iters, T, RH, cell_id: np.ndarray, kappa, rtol,
                        RH_range=(0, 1)):
        r_wet = np.empty_like(r_dry)
        for i in numba.prange(len(r_dry)):
            cid = cell_id[i]

            # root-finding initial guess
            a = r_dry[i]
            b = r_cr(kappa[i], r_dry[i] ** 3, T[cid], const.sgm_w)

            if not a < b:
                warn(msg="dry radius larger than critical radius", file=__file__,
                     context=("i", i, "r_d", r_dry[i], "T", T[cid], "RH", RH[cid], "f_org", f_org[i], "kappa", kappa[i]))
                iters[i] = -1
                continue

            # minimisation
            args = (
                T[cid],
                np.maximum(RH_range[0], np.minimum(RH_range[1], RH[cid])),
                kappa[i],
                r_dry[i]**3,
                f_org[i]
            )

            fa = minfun(a, *args)
            if fa < 0:
                r_wet[i] = r_dry[i]
                iters[i] = 0
                continue
            fb = minfun(b, *args)

            r_wet[i], iters[i] = toms748_solve(minfun, args, a, b, fa, fb, rtol=rtol, max_iter=max_iters,
                                               within_tolerance=within_tolerance)
            if iters[i] == -1:
                warn(msg="failed to find wet radius for particle", file=__file__,
                     context=("i", i, "r_d", r_dry[i], "T", T[cid], "RH", RH[cid], "f_org", f_org[i], "kappa", kappa[i]))
        return r_wet

    iters = np.empty_like(r_dry, dtype=int)
    r_wet = r_wet_init_impl(r_dry, iters, T, RH, cell_id, kappa, rtol)
    assert (iters != max_iters).all() and (iters != -1).all()
    return r_wet
