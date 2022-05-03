# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Lowe_et_al_2019 import Settings, Simulation, aerosol

from PySDM.initialisation.sampling import spectral_sampling
from PySDM.physics import si

from .constants import constants

assert hasattr(constants, "_pytestfixturefunction")


class TestFig2:
    @staticmethod
    @pytest.mark.parametrize(
        "aerosol, surface_tension, s_max, s_100m, n_100m",
        (
            (aerosol.AerosolMarine(), "Constant", 0.271, 0.081, 148),
            (aerosol.AerosolMarine(), "CompressedFilmOvadnevaite", 0.250, 0.075, 169),
            (aerosol.AerosolBoreal(), "Constant", 0.182, 0.055, 422),
            (aerosol.AerosolBoreal(), "CompressedFilmOvadnevaite", 0.137, 0.055, 525),
            (aerosol.AerosolNascent(), "Constant", 0.407, 0.122, 68),
            (aerosol.AerosolNascent(), "CompressedFilmOvadnevaite", 0.314, 0.076, 166),
        ),
    )
    @pytest.mark.xfail(strict=True)  # TODO #604
    # pylint: disable=redefined-outer-name,unused-argument
    def test_peak_supersaturation_and_final_concentration(
        *, constants, aerosol, surface_tension, s_max, s_100m, n_100m
    ):
        # arrange
        dt = 1 * si.s
        w = 0.32 * si.m / si.s
        z_max = 200 * si.m
        n_steps = int(z_max / w / dt)
        dz = z_max / n_steps
        settings = Settings(
            dz=dz,
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
        print(output["n_c_cm3"][i_100m], n_100m)
        np.testing.assert_approx_equal(np.nanmax(output["S_max"]), s_max, significant=2)
        np.testing.assert_approx_equal(output["S_max"][i_100m], s_100m, significant=2)
        np.testing.assert_approx_equal(output["n_c_cm3"][i_100m], n_100m, significant=2)
