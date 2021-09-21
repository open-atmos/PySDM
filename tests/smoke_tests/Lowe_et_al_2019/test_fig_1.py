from PySDM_examples.Lowe_et_al_2019 import aerosol
from PySDM.physics import si, Formulae, constants as const
from PySDM.physics.surface_tension import compressed_film
from .constants import constants
from scipy import signal
import numpy as np
import pytest

trivia = Formulae().trivia
r_wet = np.logspace(np.log(150 * si.nm), np.log(3000 * si.nm), base=np.e, num=100)
r_dry = 50 * si.nm
v_wet = trivia.volume(r_wet)
v_dry = trivia.volume(r_dry)
T=300 * si.K


class TestFig1:
    @staticmethod
    def test_bulk_surface_tension_is_sgm_w():
        # arrange
        formulae = Formulae(surface_tension='Constant')
        r_wet = np.logspace(np.log(150 * si.nm), np.log(3000 * si.nm), base=np.e, num=100)
        v_wet = formulae.trivia.volume(r_wet)

        # act
        sigma = formulae.surface_tension.sigma(np.nan, v_wet, np.nan, np.nan)

        # assert
        assert sigma == const.sgm_w

    @staticmethod
    @pytest.mark.parametrize(
        'aerosol, cutoff', (
                (aerosol.AerosolBoreal(), 560 * si.nm),
                (aerosol.AerosolMarine(), 380 * si.nm),
                (aerosol.AerosolNascent(), 500 * si.nm)
        )
    )
    def test_kink_location(constants, aerosol, cutoff):
        # arrange
        formulae = Formulae(surface_tension='CompressedFilm')

        # act
        sigma = formulae.surface_tension.sigma(np.nan, v_wet, v_dry, aerosol.aerosol_modes_per_cc[0]['f_org'])

        # assert
        cutoff_idx = (np.abs(r_wet - cutoff)).argmin()
        assert (sigma[:cutoff_idx] == compressed_film.sgm_org).all()
        assert sigma[cutoff_idx] > compressed_film.sgm_org
        assert .98 * const.sgm_w < sigma[-1] <= const.sgm_w

    @staticmethod
    @pytest.mark.parametrize(
        'aerosol, surface_tension, maximum_x, maximum_y, bimodal', (
                (aerosol.AerosolBoreal(),  'Constant', 320 * si.nm, .217, False),
                (aerosol.AerosolMarine(),  'Constant', 420 * si.nm, .164, False),
                (aerosol.AerosolNascent(), 'Constant', 360 * si.nm, .194, False),
                (aerosol.AerosolBoreal(),  'CompressedFilm', 360 * si.nm, .108, True),
                (aerosol.AerosolMarine(),  'CompressedFilm', 600 * si.nm, .115, False),
                (aerosol.AerosolNascent(), 'CompressedFilm', 670 * si.nm, .104, True)
        )
    )
    def test_koehler_maxima(constants, aerosol, surface_tension, maximum_x, maximum_y, bimodal):
        # arrange
        label = {'CompressedFilm': 'film', 'Constant': 'bulk'}[surface_tension]
        formulae = Formulae(surface_tension=surface_tension)
        sigma = formulae.surface_tension.sigma(np.nan, v_wet, v_dry, aerosol.aerosol_modes_per_cc[0]['f_org'])
        RH_eq = formulae.hygroscopicity.RH_eq(r_wet, T, aerosol.aerosol_modes_per_cc[0]['kappa'][label], r_dry**3, sigma)

        # act
        peaks = signal.find_peaks(RH_eq)

        # assert
        assert np.argmax(RH_eq) == (np.abs(r_wet - maximum_x)).argmin()
        np.testing.assert_approx_equal((np.amax(RH_eq)-1)*100, maximum_y, significant=3)
        assert len(peaks) == 2 if bimodal else 1
