# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM_examples.Abdul_Razzak_Ghan_2000 import data_from_ARG2000_paper as ARG_paper
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

        idx = np.argmin(np.abs(ARG_paper.Fig1_N2_param - N2i * si.cm**3))
        output = run_parcel(w, sol2, N2i, rad2, n_sd_per_mode)

        assert np.isclose(
            output.activated_fraction_S[0],
            ARG_paper.Fig1_AF_param[idx],
            atol=output.error[0] * 2,
        )
        assert np.isclose(
            output.activated_fraction_V[0],
            ARG_paper.Fig1_AF_param[idx],
            atol=output.error[0] * 2,
        )

    @staticmethod
    @pytest.mark.parametrize("N2i", np.linspace(100, 5000, 5) / si.cm**3)
    def test_ARG_fig2a(N2i):
        w = 0.5 * si.m / si.s
        sol2 = 0.1  # 10% ammonium sulfate, 90% insoluble
        rad2 = 50.0 * si.nm

        n_sd_per_mode = 10

        idx = np.argmin(np.abs(ARG_paper.Fig2a_N2_param - N2i * si.cm**3))
        output = run_parcel(w, sol2, N2i, rad2, n_sd_per_mode)

        assert np.isclose(
            output.activated_fraction_S[0],
            ARG_paper.Fig2a_AF_param[idx],
            atol=output.error[0] * 2,
        )
        assert np.isclose(
            output.activated_fraction_V[0],
            ARG_paper.Fig2a_AF_param[idx],
            atol=output.error[0] * 2,
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

        idx = np.argmin(np.abs(ARG_paper.Fig3a_sol2_param - sol2i))
        output = run_parcel(w, sol2i, N2, rad2, n_sd_per_mode)

        assert np.isclose(
            output.activated_fraction_S[0],
            ARG_paper.Fig3a_AF_param[idx],
            atol=output.error[0] * 2,
        )
        assert np.isclose(
            output.activated_fraction_V[0],
            ARG_paper.Fig3a_AF_param[idx],
            atol=output.error[0] * 2,
        )
