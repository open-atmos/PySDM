# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Lowe_et_al_2019 import Settings, Simulation
from PySDM_examples.Lowe_et_al_2019 import aerosol as paper_aerosol
from PySDM_examples.Lowe_et_al_2019.constants_def import LOWE_CONSTS

from PySDM import Formulae
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.physics import si

FORMULAE = Formulae(constants=LOWE_CONSTS)
WATER_MOLAR_VOLUME = FORMULAE.constants.water_molar_volume


class TestFig2:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize(
        "aerosol, surface_tension, s_max, s_100m, n_100m",
        (
            (
                paper_aerosol.AerosolMarine(water_molar_volume=WATER_MOLAR_VOLUME),
                "Constant",
                0.271,
                0.081,
                148,
            ),
            (
                paper_aerosol.AerosolMarine(water_molar_volume=WATER_MOLAR_VOLUME),
                "CompressedFilmOvadnevaite",
                0.250,
                0.075,
                169,
            ),
            (  # TODO #1247 SS_max & SS_100m & Nc_100m doesn't match for this case
                paper_aerosol.AerosolBoreal(water_molar_volume=WATER_MOLAR_VOLUME),
                "Constant",
                0.182,
                0.055,
                422,
            ),
            (  # TODO #1247 SS_100m & Nc_100m doesn't match for this case
                paper_aerosol.AerosolBoreal(water_molar_volume=WATER_MOLAR_VOLUME),
                "CompressedFilmOvadnevaite",
                0.137,
                0.055,
                525,
            ),
            (  # TODO #1247 SS_100m & Nc_100m doesn't match for this case
                paper_aerosol.AerosolNascent(water_molar_volume=WATER_MOLAR_VOLUME),
                "Constant",
                0.407,
                0.122,
                68,
            ),
            (  # TODO #1247 SS_100m & Nc_100m doesn't match for this case
                paper_aerosol.AerosolNascent(water_molar_volume=WATER_MOLAR_VOLUME),
                "CompressedFilmOvadnevaite",
                0.314,
                0.076,
                166,
            ),
        ),
    )
    # TODO #1247 AerosolMarine passes, but others fail
    # TODO #1246 general mismatches in parcel profiles
    @pytest.mark.xfail()
    def test_peak_supersaturation_and_final_concentration(
        *, aerosol, surface_tension, s_max, s_100m, n_100m
    ):
        # arrange
        settings = Settings(
            dz=1 * si.m,
            n_sd_per_mode=32,
            model=surface_tension,
            aerosol=aerosol,
            spectral_sampling=spectral_sampling.ConstantMultiplicity,
        )
        settings.output_interval = 10 * settings.dt
        simulation = Simulation(settings)

        # act
        output = simulation.run()

        # assert
        i_100m = np.argmin(np.abs(np.asarray(output["z"]) - 100 * si.m))
        print(i_100m, output["z"][i_100m])
        print(np.nanmax(output["S_max"]), s_max)
        print(output["S_max"][i_100m], s_100m)
        print(output["CDNC_cm3"][i_100m], n_100m)
        np.testing.assert_approx_equal(np.nanmax(output["S_max"]), s_max, significant=2)
        np.testing.assert_approx_equal(output["S_max"][i_100m], s_100m, significant=2)
        np.testing.assert_approx_equal(
            output["CDNC_cm3"][i_100m], n_100m, significant=2
        )
