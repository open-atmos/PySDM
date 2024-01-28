"""
surface tension coefficient model featuring surface-partitioning
 as in [Ruehl et al. (2016)](https://doi.org/10.1126/science.aad4889)
"""

import numpy as np


class SzyszkowskiLangmuir:  # pylint: disable=too-few-public-methods
    """
    Szyszkowski-Langmuir surface-partitioning of organics described in Ruehl et al. (2016).
    Described in supplementary materials equations (12) and (14).

    Allows for more realistic thermodynamic partitioning of some organic to the surface,
    while some remains dissolved in the bulk phase. The surface concentration is solved
    implicitly from the isotherm equation that relates the bulk organic concentration
    `C_bulk` to the surface average molecular area `A`. The equation of state relates
    the surface concentration to the surface tension.
    """

    def __init__(self, const):
        assert np.isfinite(const.RUEHL_nu_org)
        assert np.isfinite(const.RUEHL_A0)
        assert np.isfinite(const.RUEHL_C0)
        assert np.isfinite(const.RUEHL_sgm_min)

    @staticmethod
    def sigma(const, T, v_wet, v_dry, f_org):
        # wet radius (m)
        r_wet = ((3 * v_wet) / (4 * np.pi)) ** (1 / 3)

        if f_org == 0:
            sgm = const.sgm_w
        else:
            # C_bulk is the concentration of the organic in the bulk phase
            # Cb_iso = C_bulk / (1-f_surf)
            Cb_iso = (f_org * v_dry / const.RUEHL_nu_org) / (
                v_wet / const.water_molar_volume
            )

            # A is the area that one molecule of organic occupies at the droplet surface
            # A_iso = A*f_surf (m^2)
            A_iso = (4 * np.pi * r_wet**2) / (
                f_org * v_dry * const.N_A / const.RUEHL_nu_org
            )

            # fraction of organic at surface
            # quadratic formula, solve equation of state analytically
            a = -const.RUEHL_A0 / A_iso
            b = (
                const.RUEHL_A0 / A_iso
                + (const.RUEHL_A0 / A_iso) * (const.RUEHL_C0 / Cb_iso)
                + 1
            )
            c = -1
            f_surf = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

            # calculate surface tension
            sgm = const.sgm_w - (
                (const.R_str * T) / (const.RUEHL_A0 * const.N_A)
            ) * np.log(1 + Cb_iso * (1 - f_surf) / const.RUEHL_C0)

        # surface tension bounded between sgm_min and sgm_w
        sgm = min(max(sgm, const.RUEHL_sgm_min), const.sgm_w)
        return sgm


SzyszkowskiLangmuir.sigma.__vectorize = True
