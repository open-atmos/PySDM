# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning

from PySDM import Formulae
from PySDM.physics import si


def test_surface_tension(plot=False):
    ls = ["-", ":", "--", "-."]

    st_type = (
        "Constant",
        "CompressedFilmOvadnevaite",
        "SzyszkowskiLangmuir",
        "CompressedFilmRuehl",
    )
    # Arrange
    TRIVIA = Formulae().trivia
    R_WET = np.logspace(np.log(100 * si.nm), np.log(1000 * si.nm), base=np.e, num=100)
    R_DRY = 50 * si.nm
    V_WET = TRIVIA.volume(R_WET)
    V_DRY = TRIVIA.volume(R_DRY)
    TEMPERATURE = 300 * si.K

    ################
    # test zero organic
    ################

    F_ORG = 0.0
    sgm = calc_sigma(st_type, TEMPERATURE, V_WET, V_DRY, F_ORG)

    for i, st in enumerate(st_type):
        np.testing.assert_allclose(sgm[0, :], sgm[i, :], rtol=1e-6)
        plt.plot(R_WET / si.um, sgm[i, :], ls[i], label=st)

    if plot:
        plt.title(f"Forg = {F_ORG}")
        plt.xlabel("wet radius (um)")
        plt.xscale("log")
        plt.ylabel("surface tension [N/m]")
        plt.legend()
        plt.show()

    ################
    # test all organic
    ################

    F_ORG = 1.0
    sgm = calc_sigma(st_type, TEMPERATURE, V_WET, V_DRY, F_ORG)

    for i, st in enumerate(st_type):
        if i > 0:
            np.testing.assert_array_less(sgm[i, :], sgm[0, :])
        plt.plot(R_WET / si.um, sgm[i, :], ls[i], label=st)

    if plot:
        plt.title(f"Forg = {F_ORG}")
        plt.xlabel("wet radius (um)")
        plt.xscale("log")
        plt.ylabel("surface tension [N/m]")
        plt.legend()
        plt.show()


def calc_sigma(st_type, TEMPERATURE, V_WET, V_DRY, F_ORG):
    sgm = np.zeros((len(st_type), len(V_WET)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
        for i, st in enumerate(st_type):
            f = Formulae(
                surface_tension=st,
                constants={
                    "sgm_org": 10 * si.mN / si.m,
                    "delta_min": 1 * si.nm,
                    "RUEHL_A0": 1e-17 * si.m * si.m,
                    "RUEHL_C0": 1e-8,
                    "RUEHL_m_sigma": 1e17 * si.J / si.m**2,
                    "RUEHL_sgm_min": 10 * si.mN / si.m,
                    "RUEHL_nu_org": 1e2 * si.cm**3 / si.mole,
                },
            )

            for j, vw in enumerate(V_WET):
                sgm[i, j] = f.surface_tension.sigma(TEMPERATURE, vw, V_DRY, F_ORG)

    return sgm
