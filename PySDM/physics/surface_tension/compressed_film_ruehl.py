import numpy as np
from scipy import constants as sci
from scipy import optimize
from PySDM.physics import constants as const

nu_org = np.nan
A0 = np.nan
C0 = np.nan
m_sigma = np.nan
sgm_min = np.nan


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
    @staticmethod
    def _check():
        assert np.isfinite(nu_org)
        assert np.isfinite(A0)
        assert np.isfinite(C0)
        assert np.isfinite(m_sigma)
        assert np.isfinite(sgm_min)

    @staticmethod
    def sigma(T, v_wet, v_dry, f_org):
        r_wet = ((3 * v_wet) / (4 * np.pi))**(1/3) # m - wet radius

        # C_bulk is the concentration of the organic in the bulk phase
        Cb_iso = (f_org*v_dry/nu_org) / (v_wet/const.nu_w) # = C_bulk / (1-f_surf)

        # A is the area that one molecule of organic occupies at the droplet surface
        A_iso = (4 * np.pi * r_wet**2) / (f_org * v_dry * sci.N_A / nu_org) # m^2 = A*f_surf

        # solve implicitly for fraction of organic at surface
        f = lambda f_surf: Cb_iso*(1-f_surf) - C0*np.exp(
            ((A0**2 - (A_iso/f_surf)**2)*m_sigma*sci.N_A)/(2*sci.R*T)
        )
        sol = optimize.root_scalar(f, bracket=[0,1])
        f_surf = sol.root

        # calculate surface tension
        sgm = const.sgm_w - (A0 - A_iso/f_surf)*m_sigma # m^2 * J/m^2 = J = N*m --> N/m - N*m ?
        sgm = np.minimum(np.maximum(sgm, sgm_min), const.sgm_w)
        return sgm
