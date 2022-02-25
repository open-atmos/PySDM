# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
import numpy as np
from PySDM_examples.Lowe_et_al_2019 import Settings, Simulation, aerosol
from PySDM.initialisation.sampling import spectral_sampling
from PySDM.physics import si
from .constants import constants

assert hasattr(constants, '_pytestfixturefunction')


class TestFig2:
    @staticmethod
    @pytest.mark.parametrize("aerosol, surface_tension, s_max, s_100m, n_100m", (
        (aerosol.AerosolMarine(),  "Constant", .271, .081, 148),
        (aerosol.AerosolMarine(),  "CompressedFilmOvadnevaite", .250, .075, 169),
        (aerosol.AerosolBoreal(),  "Constant", .182, .055, 422),
        (aerosol.AerosolBoreal(),  "CompressedFilmOvadnevaite", .137, .055, 525),
        (aerosol.AerosolNascent(), "Constant", .407, .122, 68),
        (aerosol.AerosolNascent(), "CompressedFilmOvadnevaite", .314, .076, 166)
    ))
    @pytest.mark.xfail(strict=True)  # TODO #604
    # pylint: disable=redefined-outer-name,unused-argument
    def test_peak_supersaturation_and_final_concentration(
        constants, aerosol, surface_tension, s_max, s_100m, n_100m
    ):
        # arrange
        settings = Settings(
            dz=2/.32 * si.m,
            n_sd_per_mode=32,
            model={'CompressedFilmOvadnevaite': 'film', 'Constant': 'bulk'}[surface_tension],
            aerosol=aerosol,
            spectral_sampling=spectral_sampling.ConstantMultiplicity
        )
        settings.output_interval = 10 * settings.dt
        simulation = Simulation(settings)

        # act
        output = simulation.run()

        # assert
        # assert len(output['S_max']) == 2
        i_100m = 312
        #print(output["z"][i_100m])
        print(np.nanmax(output['S_max']), s_max)
        print(output['S_max'][i_100m], s_100m)
        print(output['n_c_cm3'][i_100m], n_100m)
        np.testing.assert_approx_equal(np.nanmax(output['S_max']), s_max, significant=2)
        np.testing.assert_approx_equal(output['S_max'][i_100m], s_100m, significant=2)
        np.testing.assert_approx_equal(output['n_c_cm3'][i_100m], n_100m, significant=2)
