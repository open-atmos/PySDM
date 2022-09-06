"""
surface tension coefficient model featuring surface-partitioning
 as in [Ruehl et al. (2016)](https://doi.org/10.1126/science.aad4889)
"""
import numba
import numpy as np

from PySDM.backends.impl_numba.conf import JIT_FLAGS as jit_flags
from PySDM.backends.impl_numba.toms748 import toms748_solve
from PySDM.physics.trivia import Trivia


# pylint: disable=too-many-arguments
@numba.njit(**{**jit_flags, "parallel": False})
def minfun(f_surf, Cb_iso, RUEHL_C0, RUEHL_A0, A_iso, c):
    lhs = Cb_iso * (1 - f_surf) / RUEHL_C0
    rhs = np.exp(c * (RUEHL_A0**2 - (A_iso / f_surf) ** 2))
    return lhs - rhs


within_tolerance = numba.njit(
    Trivia.within_tolerance, **{**jit_flags, "parallel": False}
)


class CompressedFilmRuehl:
    """
    Compressed film model of surface-partitioning of organics from Ruehl et al. (2016).
    Described in supplementary materials equations (13) and (15).

    Allows for more realistic thermodynamic partitioning of some organic to the surface,
    while some remains dissolved in the bulk phase. The surface concentration is solved
    implicitly from the isotherm equation that relates the bulk organic concentration
    `C_bulk` to the surface average molecular area `A`. The equation of state relates
    the surface concentration to the surface tension. For the compressed film model it
    is linear, with slope `m_sigma`.
    """

    def __init__(self, const):
        assert np.isfinite(const.RUEHL_nu_org)
        assert np.isfinite(const.RUEHL_A0)
        assert np.isfinite(const.RUEHL_C0)
        assert np.isfinite(const.RUEHL_m_sigma)
        assert np.isfinite(const.RUEHL_sgm_min)

    @staticmethod
    def sigma(const, T, v_wet, v_dry, f_org):
        # wet radius (m)
        r_wet = ((3 * v_wet) / (4 * const.PI)) ** (1 / 3)

        zero = (
            np.equal(f_org, np.zeros_like(f_org))
            * np.equal(v_wet, v_wet)
            * np.equal(v_dry, v_dry)
            * np.ones(1)
        )
        f_org = np.where(zero, 0.5, f_org)

        # C_bulk is the concentration of the organic in the bulk phase
        # Cb_iso = C_bulk / (1-f_surf)
        Cb_iso = (f_org * v_dry / const.RUEHL_nu_org) / (v_wet / const.nu_w)

        # A is the area one molecule of organic occupies at the droplet surface
        # A_iso = A*f_surf (m^2)
        A_iso = (4 * const.PI * r_wet**2) / (
            f_org * v_dry * const.N_A / const.RUEHL_nu_org
        )

        # solve implicitly for fraction of organic at surface
        c = (const.RUEHL_m_sigma * const.N_A) / (2 * const.R_str * T)

        # loop through
        v_wet = v_wet * np.ones_like(f_org)
        v_dry = v_dry * np.ones_like(f_org)
        f_surf = np.zeros_like(f_org)
        for i, f in enumerate(f_org):
            args = (Cb_iso[i], const.RUEHL_C0, const.RUEHL_A0, A_iso[i], c)
            rtol = 1e-6
            max_iters = 1e2
            bracket = (1e-16, 1)
            f0 = minfun(0, *args)
            f1 = minfun(bracket[0], *args)
            f2 = minfun(bracket[1], *args)
            if (f1 * f2 > 0) & (f0 * f2 < 0):
                f_surf[i] = 0
                iters = 0
                print("error", f0, f1, f2)
            else:
                f_surf[i], iters = toms748_solve(
                    minfun, args, *bracket, f1, f2, rtol, max_iters, within_tolerance
                )
            assert iters != max_iters

        # calculate surface tension
        _sgm = const.sgm_w - (const.RUEHL_A0 - A_iso / f_surf) * const.RUEHL_m_sigma
        _sgm_w = const.sgm_w * np.ones_like(_sgm)
        sgm = np.where(zero, _sgm_w, _sgm)

        # surface tension bounded between sgm_min and sgm_w
        sgm = np.minimum(np.maximum(sgm, const.RUEHL_sgm_min), const.sgm_w)
        return sgm


CompressedFilmRuehl.sigma.__extras = {
    "toms748_solve": toms748_solve,
    "minfun": minfun,
    "within_tolerance": within_tolerance,
}
