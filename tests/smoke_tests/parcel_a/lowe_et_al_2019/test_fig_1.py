# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Lowe_et_al_2019 import aerosol as paper_aerosol
from PySDM_examples.Lowe_et_al_2019.constants_def import LOWE_CONSTS
from scipy import signal

from PySDM import Formulae
from PySDM.physics import constants_defaults as const
from PySDM.physics import si

FORMULAE = Formulae(constants=LOWE_CONSTS)
TRIVIA = FORMULAE.trivia
R_WET = np.logspace(np.log(150 * si.nm), np.log(3000 * si.nm), base=np.e, num=100)
R_DRY = 50 * si.nm
V_WET = TRIVIA.volume(R_WET)
V_DRY = TRIVIA.volume(R_DRY)
TEMPERATURE = 300 * si.K
WATER_MOLAR_VOLUME = FORMULAE.constants.water_molar_volume


class TestFig1:
    @staticmethod
    def test_bulk_surface_tension_is_sgm_w():
        # arrange
        formulae = Formulae(surface_tension="Constant", constants=LOWE_CONSTS)
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
            (
                paper_aerosol.AerosolBoreal(water_molar_volume=WATER_MOLAR_VOLUME),
                560 * si.nm,
            ),
            (
                paper_aerosol.AerosolMarine(water_molar_volume=WATER_MOLAR_VOLUME),
                380 * si.nm,
            ),
            (
                paper_aerosol.AerosolNascent(water_molar_volume=WATER_MOLAR_VOLUME),
                500 * si.nm,
            ),
        ),
    )
    # pylint: disable=unused-argument
    def test_kink_location(constants, aerosol, cutoff):
        # arrange
        formulae = Formulae(
            surface_tension="CompressedFilmOvadnevaite",
            constants=LOWE_CONSTS,
        )

        # act
        sigma = formulae.surface_tension.sigma(
            np.nan, V_WET, V_DRY, aerosol.modes[0]["f_org"]
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
            (
                paper_aerosol.AerosolBoreal(water_molar_volume=WATER_MOLAR_VOLUME),
                "Constant",
                320 * si.nm,
                0.217,
                False,
            ),
            (
                paper_aerosol.AerosolMarine(water_molar_volume=WATER_MOLAR_VOLUME),
                "Constant",
                420 * si.nm,
                0.164,
                False,
            ),
            (
                paper_aerosol.AerosolNascent(water_molar_volume=WATER_MOLAR_VOLUME),
                "Constant",
                360 * si.nm,
                0.194,
                False,
            ),
            (
                paper_aerosol.AerosolBoreal(water_molar_volume=WATER_MOLAR_VOLUME),
                "CompressedFilmOvadnevaite",
                360 * si.nm,
                0.108,
                True,
            ),
            (
                paper_aerosol.AerosolMarine(water_molar_volume=WATER_MOLAR_VOLUME),
                "CompressedFilmOvadnevaite",
                600 * si.nm,
                0.115,
                False,
            ),
            (
                paper_aerosol.AerosolNascent(water_molar_volume=WATER_MOLAR_VOLUME),
                "CompressedFilmOvadnevaite",
                670 * si.nm,
                0.104,
                True,
            ),
        ),
    )
    def test_koehler_maxima(*, aerosol, surface_tension, maximum_x, maximum_y, bimodal):
        # arrange
        formulae = Formulae(
            surface_tension=surface_tension,
            constants=LOWE_CONSTS,
        )
        sigma = formulae.surface_tension.sigma(
            np.nan, V_WET, V_DRY, aerosol.modes[0]["f_org"]
        )
        RH_eq = formulae.hygroscopicity.RH_eq(
            R_WET,
            TEMPERATURE,
            aerosol.modes[0]["kappa"][surface_tension],
            R_DRY**3,
            sigma,
        )

        # act
        peaks, _ = signal.find_peaks(RH_eq)

        # assert
        assert np.argmax(RH_eq) == (np.abs(R_WET - maximum_x)).argmin()
        np.testing.assert_approx_equal(
            (np.amax(RH_eq) - 1) * 100, maximum_y, significant=3
        )
        assert len(peaks) == 2 if bimodal else 1
