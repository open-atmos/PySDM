# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from PySDM_examples.Shipway_and_Hill_2012 import Settings


class TestSettings:
    @staticmethod
    def test_instantiate():
        _ = Settings(n_sd_per_gridbox=1, rho_times_w_1=1)

    @staticmethod
    def test_th():
        settings = Settings(n_sd_per_gridbox=1, rho_times_w_1=1)
        assert settings._th(0) == 297.9
        assert settings._th(100) == 297.9
        assert settings._th(740) == 297.9
        assert settings._th(3260) == 312.66

    @staticmethod
    def test_water_vapour_mixing_ratio():
        settings = Settings(n_sd_per_gridbox=1, rho_times_w_1=1)
        assert settings.water_vapour_mixing_ratio(0) == 0.015
        assert settings.water_vapour_mixing_ratio(740) == 0.0138
        np.testing.assert_approx_equal(settings.water_vapour_mixing_ratio(3260), 0.0024)

    @staticmethod
    def test_rhod():
        settings = Settings(n_sd_per_gridbox=1, rho_times_w_1=1)
        assert settings.rhod
