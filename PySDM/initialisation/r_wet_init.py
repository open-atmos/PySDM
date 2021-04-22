"""
Crated at 2019
"""

import numpy as np
from ..backends.numba.toms748 import toms748_solve
from ..physics import formulae as phys
from ..physics import constants as const
from ..backends.numba.conf import JIT_FLAGS
from numba import prange, njit

default_rtol = 1e-5


def r_wet_init(r_dry: np.ndarray, environment, cell_id: np.ndarray, kappa, rtol=default_rtol):
    T = environment["T"].to_ndarray()
    p = environment["p"].to_ndarray()
    RH = environment["RH"].to_ndarray()

    formulae = environment.core.backend.formulae
    pvs_C = formulae.saturation_vapour_pressure.pvs_Celsius
    lv_K = formulae.latent_heat.lv
    phys_dr_dt = formulae.drop_growth.dr_dt

    @njit(**{**JIT_FLAGS, **{'parallel': False, 'fastmath': False, 'cache': False}})
    def minfun(r, T, p, RH, lv, pvs, kp, rd):
        RH_eq = phys.RH_eq(r, T, kp, rd)
        D = phys.D(r, T)
        K = phys.K(r, T, p)
        return phys_dr_dt(r, RH_eq, T, RH, lv, pvs, D, K)

    return r_wet_init_impl(pvs_C, lv_K, minfun, r_dry, T, p, RH, cell_id, kappa, rtol)


@njit(**{**JIT_FLAGS, **{'parallel': False, 'fastmath': False, 'cache': False}})
def r_wet_init_impl(pvs_C, lv_K, minfun, r_dry: np.ndarray, T, p, RH, cell_id: np.ndarray, kappa, rtol, RH_range=(0, 1)):
    r_wet = np.empty_like(r_dry)
    lv = lv_K(T)
    pvs = pvs_C(T - const.T0)
    for i in prange(len(r_dry)):
        r_d = r_dry[i]
        cid = cell_id[i]
        # root-finding initial guess
        a = r_d
        b = phys.r_cr(kappa, r_d, T[cid])
        # minimisation
        args = (
            T[cid],
            p[cid],
            np.maximum(RH_range[0], np.minimum(RH_range[1], RH[cid])),
            lv[cid],
            pvs[cid],
            kappa,
            r_d
        )
        fa = minfun(a, *args)
        fb = minfun(b, *args)
        max_iters = 64
        r_wet[i], iters_done = toms748_solve(minfun, args, a, b, fa, fb, rtol=rtol, max_iter=max_iters)
        assert iters_done != max_iters
    return r_wet
