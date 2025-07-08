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

# pylint: disable=too-many-locals,too-many-arguments


def _solve_equilibrium_radii(
    *,
    radii_in: np.ndarray,
    get_bounds,
    get_args,
    minfun,
    environment,
    kappa,
    f_org: np.ndarray,
    cell_id: np.ndarray,
    rtol: float,
    max_iters: int,
    skip_fa_lt_zero: bool,
    RH_range: tuple = (0, 1),
):
    T = environment["T"].to_ndarray()
    RH = environment["RH"].to_ndarray()
    formulae = environment.particulator.formulae
    within_tolerance = formulae.trivia.within_tolerance
    jit_flags = {**JIT_FLAGS, **{"fastmath": formulae.fastmath}}

    if cell_id is None:
        cell_id = np.zeros_like(radii_in, dtype=int)
    if f_org is None:
        f_org = np.zeros_like(radii_in, dtype=float)

    @numba.njit(**jit_flags)
    def impl(radii_in, iters, T, RH, cell_id, kappa, f_org):
        radii_out = np.empty_like(radii_in)
        for i in numba.prange(len(radii_in)):  # pylint: disable=not-an-iterable
            cid = cell_id[i]
            a, b = get_bounds(radii_in[i], T[cid], kappa[i])

            if not a < b:
                radii_out[i] = radii_in[i]
                iters[i] = 0
                continue

            RH_i = np.maximum(RH_range[0], np.minimum(RH_range[1], RH[cid]))
            args = get_args(T[cid], RH_i, kappa[i], radii_in[i], f_org[i])
            fa, fb = minfun(a, *args), minfun(b, *args)

            if skip_fa_lt_zero and fa < 0:
                radii_out[i] = radii_in[i]
                iters[i] = 0
                continue

            radii_out[i], iters[i] = toms748_solve(
                minfun,
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
                    msg="failed to find equilibrium particle size",
                    file=__file__,
                    context=(
                        "i",
                        i,
                        "r",
                        radii_in[i],
                        "T",
                        T[cid],
                        "RH",
                        RH_i,
                        "f_org",
                        f_org[i],
                        "kappa",
                        kappa[i],
                    ),
                )
        return radii_out

    iters = np.empty_like(radii_in, dtype=int)
    radii_out = impl(radii_in, iters, T, RH, cell_id, kappa, f_org)
    assert (iters != max_iters).all() and (iters != -1).all()
    return radii_out


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
    formulae = environment.particulator.formulae
    sigma = formulae.surface_tension.sigma
    phys_volume = formulae.trivia.volume
    const = environment.particulator.formulae.constants
    RH_eq = formulae.hygroscopicity.RH_eq
    jit_flags = {**JIT_FLAGS, **{"fastmath": formulae.fastmath}}

    @numba.njit(**{**jit_flags, "parallel": False})
    def get_args(T_i, RH_i, kappa, r_wet, f_org):
        return T_i, RH_i, kappa, r_wet, f_org

    @numba.njit(**{**jit_flags, "parallel": False})
    def get_bounds(r_wet, _, __):
        return 0.0, r_wet

    @numba.njit(**{**jit_flags, "parallel": False})
    def minfun_dry(r_dry, temperature, relative_humidity, kappa, r_wet, f_org):
        r_dry_3 = r_dry**3
        sgm = sigma(
            temperature, phys_volume(radius=r_wet), const.PI_4_3 * r_dry_3, f_org
        )
        return relative_humidity - RH_eq(r_wet, temperature, kappa, r_dry_3, sgm)

    return _solve_equilibrium_radii(
        radii_in=r_wet,
        get_args=get_args,
        get_bounds=get_bounds,
        minfun=minfun_dry,
        environment=environment,
        kappa=kappa,
        f_org=f_org,
        cell_id=cell_id,
        rtol=rtol,
        max_iters=max_iters,
        skip_fa_lt_zero=False,
    )


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
    formulae = environment.particulator.formulae
    sigma = formulae.surface_tension.sigma
    phys_volume = formulae.trivia.volume
    const = environment.particulator.formulae.constants
    RH_eq = formulae.hygroscopicity.RH_eq
    r_cr = formulae.hygroscopicity.r_cr
    jit_flags = {**JIT_FLAGS, **{"fastmath": formulae.fastmath}}

    @numba.njit(**{**jit_flags, "parallel": False})
    def get_args(T_i, RH_i, kappa, r_dry, f_org):
        return T_i, RH_i, kappa, r_dry**3, f_org

    @numba.njit(**{**jit_flags, "parallel": False})
    def get_bounds(r_dry, T_i, kappa):
        a = r_dry
        b = r_cr(kappa, r_dry**3, T_i, const.sgm_w)
        return a, b

    @numba.njit(**{**jit_flags, "parallel": False})
    def minfun_wet(r_wet, temperature, relative_humidity, kappa, r_dry_3, f_org):
        sgm = sigma(
            temperature, phys_volume(radius=r_wet), const.PI_4_3 * r_dry_3, f_org
        )
        return relative_humidity - RH_eq(r_wet, temperature, kappa, r_dry_3, sgm)

    return _solve_equilibrium_radii(
        radii_in=r_dry,
        get_args=get_args,
        get_bounds=get_bounds,
        minfun=minfun_wet,
        environment=environment,
        kappa=kappa_times_dry_volume / phys_volume(radius=r_dry),
        f_org=f_org,
        cell_id=cell_id,
        rtol=rtol,
        max_iters=max_iters,
        skip_fa_lt_zero=True,
    )
