# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Pyrcel import Settings, Simulation

from PySDM import Formulae
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si
from PySDM.products import (
    AmbientRelativeHumidity,
    AmbientTemperature,
    ParcelDisplacement,
)


class TestParcelExample:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize("s_max, s_250m, T_250m", ((0.62, 0.139, 272.2),))
    @pytest.mark.parametrize("scipy_solver", (pytest.param(True), pytest.param(False)))
    @pytest.mark.xfail(
        strict=True
    )  # TODO #1246 s_250m (only) fails for both solver options
    def test_supersaturation_and_temperature_profile(
        s_max, s_250m, T_250m, scipy_solver
    ):
        # arrange
        settings = Settings(
            dz=1 * si.m,
            n_sd_per_mode=(5, 5),
            aerosol_modes_by_kappa={
                0.54: Lognormal(
                    norm_factor=850 / si.cm**3, m_mode=15 * si.nm, s_geom=1.6
                ),
                1.2: Lognormal(
                    norm_factor=10 / si.cm**3, m_mode=850 * si.nm, s_geom=1.2
                ),
            },
            vertical_velocity=1.0 * si.m / si.s,
            initial_pressure=775 * si.mbar,
            initial_temperature=274 * si.K,
            initial_relative_humidity=0.98,
            displacement=250 * si.m,
            formulae=Formulae(constants={"MAC": 0.3}),
        )
        simulation = Simulation(
            settings,
            products=(
                ParcelDisplacement(name="z"),
                AmbientRelativeHumidity(name="RH", unit="%"),
                AmbientTemperature(name="T"),
            ),
            scipy_solver=scipy_solver,
        )

        # act
        output = simulation.run()

        # assert
        print(np.nanmax(np.asarray(output["products"]["RH"])) - 100, s_max)
        print(output["products"]["T"][-1], T_250m)
        print(output["products"]["RH"][-1] - 100, s_250m)
        np.testing.assert_approx_equal(
            np.nanmax(np.asarray(output["products"]["RH"])) - 100, s_max, significant=2
        )
        np.testing.assert_approx_equal(
            output["products"]["T"][-1], T_250m, significant=2
        )
        np.testing.assert_approx_equal(
            output["products"]["RH"][-1] - 100, s_250m, significant=2
        )
