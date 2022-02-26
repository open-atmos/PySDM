# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
import numpy as np
from PySDM_examples.Pyrcel import Settings, Simulation
from PySDM import Formulae
from PySDM.initialisation.spectra import Lognormal
from PySDM.physics import si
from PySDM.products import (ParcelDisplacement, PeakSupersaturation, AmbientTemperature,)


class TestParcelExample:
    @staticmethod
    @pytest.mark.parametrize("s_max, s_250m, T_250m", (
        (0.62,0.139,272.2),
    ))
    @pytest.mark.xfail(strict=True)  # TODO #776
    # pylint: disable=redefined-outer-name,unused-argument
    def test_supersaturation_and_temperature_profile(s_max, s_250m, T_250m):
        # arrange
        settings = Settings(
            dz = 1 * si.m,
            n_sd_per_mode = (5, 5),
            aerosol_modes_by_kappa = {
                .54: Lognormal(
                    norm_factor=850 / si.cm ** 3,
                    m_mode=15 * si.nm,
                    s_geom=1.6
                ),
                1.2: Lognormal(
                    norm_factor=10 / si.cm ** 3,
                    m_mode=850 * si.nm,
                    s_geom=1.2
                )
            },
            vertical_velocity = 1.0 * si.m / si.s,
            initial_pressure = 775 * si.mbar,
            initial_temperature = 274 * si.K,
            initial_relative_humidity = .98,
            displacement = 250 * si.m,
            formulae = Formulae(constants={'MAC': .3})
        )
        simulation = Simulation(settings, products=(
            ParcelDisplacement(
                name='z'),
            PeakSupersaturation(
                name='S_max', unit='%'),
            AmbientTemperature(
                name='T'),
        ))

        # act
        output = simulation.run()

        # assert
        np.testing.assert_approx_equal(np.nanmax(output['products']['S_max']), s_max, significant=2)
        np.testing.assert_approx_equal(output['products']['S_max'][-1], s_250m, significant=2)
        np.testing.assert_approx_equal(output['products']['T'][-1], T_250m, significant=2)
