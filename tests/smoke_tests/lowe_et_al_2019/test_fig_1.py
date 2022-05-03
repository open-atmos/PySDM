# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Lowe_et_al_2019 import aerosol
from scipy import signal

from PySDM import Formulae
from PySDM.physics import constants_defaults as const
from PySDM.physics import si

from .constants import constants

assert hasattr(constants, "_pytestfixturefunction")

TRIVIA = Formulae().trivia
R_WET = np.logspace(np.log(150 * si.nm), np.log(3000 * si.nm), base=np.e, num=100)
R_DRY = 50 * si.nm
V_WET = TRIVIA.volume(R_WET)
V_DRY = TRIVIA.volume(R_DRY)
TEMPERATURE = 300 * si.K


class TestFig1:
    @staticmethod
    def test_bulk_surface_tension_is_sgm_w():
        # arrange
        formulae = Formulae(surface_tension="Constant")
        r_wet = np.logspace(
            np.log(150 * si.nm), np.log(3000 * si.nm), base=np.e, num=100
        )
        v_wet = formulae.trivia.volume(r_wet)

        # act
        sigma = formulae.surface_tension.sigma(np.nan, v_wet, np.nan, np.nan)

        # assert
        assert sigma == const.sgm_w

    @staticmethod
    @pytest.mark.parametrize(
        "aerosol, cutoff",
        (
            (aerosol.AerosolBoreal(), 560 * si.nm),
            (aerosol.AerosolMarine(), 380 * si.nm),
            (aerosol.AerosolNascent(), 500 * si.nm),
        ),
    )
    # pylint: disable=redefined-outer-name,unused-argument
    def test_kink_location(constants, aerosol, cutoff):
        # arrange
        formulae = Formulae(
            surface_tension="CompressedFilmOvadnevaite",
            constants={"sgm_org": 40 * si.mN / si.m, "delta_min": 0.1 * si.nm},
        )

        # act
        sigma = formulae.surface_tension.sigma(
            np.nan, V_WET, V_DRY, aerosol.aerosol_modes[0]["f_org"]
        )

        # assert
        cutoff_idx = (np.abs(R_WET - cutoff)).argmin()
        assert (sigma[:cutoff_idx] == formulae.constants.sgm_org).all()
        assert sigma[cutoff_idx] > formulae.constants.sgm_org
        assert 0.98 * const.sgm_w < sigma[-1] <= const.sgm_w

    @staticmethod
    @pytest.mark.parametrize(
        "aerosol, surface_tension, maximum_x, maximum_y, bimodal",
        (
            (aerosol.AerosolBoreal(), "Constant", 320 * si.nm, 0.217, False),
            (aerosol.AerosolMarine(), "Constant", 420 * si.nm, 0.164, False),
            (aerosol.AerosolNascent(), "Constant", 360 * si.nm, 0.194, False),
            (
                aerosol.AerosolBoreal(),
                "CompressedFilmOvadnevaite",
                360 * si.nm,
                0.108,
                True,
            ),
            (
                aerosol.AerosolMarine(),
                "CompressedFilmOvadnevaite",
                600 * si.nm,
                0.115,
                False,
            ),
            (
                aerosol.AerosolNascent(),
                "CompressedFilmOvadnevaite",
                670 * si.nm,
                0.104,
                True,
            ),
        ),
    )
    # pylint: disable=redefined-outer-name,unused-argument
    def test_koehler_maxima(
        *, constants, aerosol, surface_tension, maximum_x, maximum_y, bimodal
    ):
        # arrange
        formulae = Formulae(
            surface_tension=surface_tension,
            constants={"sgm_org": 40 * si.mN / si.m, "delta_min": 0.1 * si.nm},
        )
        sigma = formulae.surface_tension.sigma(
            np.nan, V_WET, V_DRY, aerosol.aerosol_modes[0]["f_org"]
        )
        RH_eq = formulae.hygroscopicity.RH_eq(
            R_WET,
            TEMPERATURE,
            aerosol.aerosol_modes[0]["kappa"][surface_tension],
            R_DRY**3,
            sigma,
        )

        # act
        peaks = signal.find_peaks(RH_eq)

        # assert
        assert np.argmax(RH_eq) == (np.abs(R_WET - maximum_x)).argmin()
        np.testing.assert_approx_equal(
            (np.amax(RH_eq) - 1) * 100, maximum_y, significant=3
        )
        assert len(peaks) == 2 if bimodal else 1
