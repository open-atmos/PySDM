# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Abdul_Razzak_Ghan_2000.data_from_ARG2000_paper import (  # Fig2a_N2_obs, Fig2a_AF_obs, Fig2a_N2_param, Fig2a_AF_param,; Fig2b_N2_obs, Fig2b_AF_obs, Fig2b_N2_param, Fig2b_AF_param,; Fig3b_sol2_obs, Fig3b_AF_obs, Fig3b_sol2_param, Fig3b_AF_param,; Fig4a_rad2_obs, Fig4a_AF_obs, Fig4a_rad2_param, Fig4a_AF_param,; Fig4b_rad2_obs, Fig4b_AF_obs, Fig4b_rad2_param, Fig4b_AF_param,; Fig5a_w_obs, Fig5a_AF_obs, Fig5a_w_param, Fig5a_AF_param,; Fig5b_w_obs, Fig5b_AF_obs, Fig5b_w_param, Fig5b_AF_param
    Fig1_AF_obs,
    Fig1_AF_param,
    Fig1_N2_obs,
    Fig1_N2_param,
    Fig3a_AF_obs,
    Fig3a_AF_param,
    Fig3a_sol2_obs,
    Fig3a_sol2_param,
)
from PySDM_examples.Abdul_Razzak_Ghan_2000.run_ARG_parcel import run_parcel

from PySDM.physics import si


class TestARGExample:  # pylint: disable=too-few-public-methods
    @staticmethod
    @pytest.mark.parametrize("N2i", np.linspace(100, 5000, 5) / si.cm**3)
    def test_ARG_fig1(N2i):
        w = 0.5 * si.m / si.s
        sol2 = 1.0  # 100% ammonium sulfate
        rad2 = 50.0 * si.nm

        n_sd_per_mode = 10

        idx = np.argmin(np.abs(Fig1_N2_param - N2i * si.cm**3))
        output = run_parcel(w, sol2, N2i, rad2, n_sd_per_mode)

        assert np.isclose(
            output.activated_fraction_S[0], Fig1_AF_param[idx], atol=output.error[0]
        )
        assert np.isclose(
            output.activated_fraction_V[0], Fig1_AF_param[idx], atol=output.error[0]
        )

    @staticmethod
    @pytest.mark.parametrize(
        "sol2i", np.linspace(0.1, 1, 5)
    )  # X% ammonium sulfate, (1-X)% insoluble
    def test_ARG_fig3a(sol2i):
        N2 = 100 / si.cm**3
        w = 0.5 * si.m / si.s
        rad2 = 50.0 * si.nm

        n_sd_per_mode = 10

        idx = np.argmin(np.abs(Fig3a_sol2_param - sol2i))
        output = run_parcel(w, sol2i, N2, rad2, n_sd_per_mode)

        assert np.isclose(
            output.activated_fraction_S[0], Fig3a_AF_param[idx], atol=output.error[0]
        )
        assert np.isclose(
            output.activated_fraction_V[0], Fig3a_AF_param[idx], atol=output.error[0]
        )
