# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM import Formulae
from PySDM.physics import constants_defaults as const
from PySDM.physics import si


def test_surface_tension(plot=False):
    st_type = (
        "Constant",
        "CompressedFilmOvadnevaite",
        "SzyszkowskiLangmuir",
        "CompressedFilmRuehl",
    )
    # Arrange
    TRIVIA = Formulae().trivia
    R_WET = np.logspace(np.log(150 * si.nm), np.log(3000 * si.nm), base=np.e, num=100)
    R_DRY = 50 * si.nm
    V_WET = TRIVIA.volume(R_WET)
    V_DRY = TRIVIA.volume(R_DRY)
    TEMPERATURE = 300 * si.K

    const.sgm_org = 10 * si.mN / si.m
    const.delta_min = 1 * si.nm
    const.RUEHL_A0 = 1e-17 * si.m * si.m
    const.RUEHL_C0 = 1e-8
    const.RUEHL_m_sigma = 1e17 * si.J / si.m**2
    const.RUEHL_sgm_min = 10 * si.mN / si.m
    const.RUEHL_nu_org = 1e2 * si.cm**3 / si.mole

    F_ORG = 0.0
    sgm = np.zeros((len(st_type), len(V_WET)))
    for i, st in enumerate(st_type):
        f = Formulae(surface_tension=st)
        for j, vw in enumerate(V_WET):
            sgm[i, j] = f.surface_tension.sigma(TEMPERATURE, vw, V_DRY, F_ORG)

        np.testing.assert_allclose(sgm[0, :], sgm[i, :], rtol=1e-2)

    F_ORG = 1.0
    sgm = np.zeros((len(st_type), len(V_WET)))
    for i, st in enumerate(st_type):
        f = Formulae(surface_tension=st)
        for j, vw in enumerate(V_WET):
            sgm[i, j] = f.surface_tension.sigma(TEMPERATURE, vw, V_DRY, F_ORG)

        np.testing.assert_array_less(sgm[i, :], sgm[0, :], rtol=1e-2)
