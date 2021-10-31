from PySDM_examples.Shipway_and_Hill_2012 import Settings, Simulation
import numpy as np


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
    def test_qv():
        settings = Settings(n_sd_per_gridbox=1, rho_times_w_1=1)
        assert settings.qv(0) == .015
        assert settings.qv(740) == .0138
        np.testing.assert_approx_equal(settings.qv(3260), .0024)

    @staticmethod
    def test_rhod():
        settings = Settings(n_sd_per_gridbox=1, rho_times_w_1=1)
        assert settings.rhod
