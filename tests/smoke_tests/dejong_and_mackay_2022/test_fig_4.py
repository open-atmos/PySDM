# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import matplotlib
import numpy as np
from matplotlib import pyplot
from PySDM_examples.deJong_Mackay_2022 import Settings0D, run_box_breakup

from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN, Gaussian
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.physics.constants import si

from ...backends_fixture import backend_class

assert hasattr(backend_class, "_pytestfixturefunction")

CMAP = matplotlib.cm.get_cmap("viridis")
N_SD = 2**12
DT = 1 * si.s


def bins_edges(num):
    return np.logspace(
        np.log10(5e0 * si.um), np.log10(5e3 * si.um), num=num, endpoint=True
    )


# pylint: disable=redefined-outer-name
class TestFig4:
    @staticmethod
    def test_fig_4a(backend_class, plot=False):
        # arrange
        settings0 = Settings0D(seed=44)
        settings0.n_sd = N_SD
        settings0.radius_bins_edges = bins_edges(32)

        nf_vals = [1, 4, 16, 64]
        data_x = {}
        data_y = {}

        # act
        lbl = "initial"
        (data_x[lbl], data_y[lbl], _) = run_box_breakup(settings0, [0], backend_class)
        for i, nf_val in enumerate(nf_vals):
            settings = Settings0D(
                fragmentation=AlwaysN(n=nf_val), seed=44, warn_overflows=False
            )
            settings.n_sd = settings0.n_sd
            settings.radius_bins_edges = settings0.radius_bins_edges
            settings.coal_eff = ConstEc(Ec=0.95)
            settings.dt = DT

            lbl = "n_f = " + str(nf_val)
            (data_x[lbl], data_y[lbl], _) = run_box_breakup(
                settings, [120], backend_class
            )

        # plot
        pyplot.step(
            data_x["initial"],
            data_y["initial"][0] * settings.rho,
            color="k",
            linestyle="--",
            label="initial",
        )
        for i, nf_val in enumerate(nf_vals):
            lbl = "n_f = " + str(nf_val)
            pyplot.step(
                data_x[lbl],
                data_y[lbl][0] * settings.rho,
                color=CMAP(i / len(nf_vals)),
                label=lbl
                if lbl not in pyplot.gca().get_legend_handles_labels()[1]
                else "",
            )
        pyplot.xscale("log")
        pyplot.xlabel("particle radius (um)")
        pyplot.ylabel("dm/dlnR (kg/m$^3$ / unit(ln R)")
        pyplot.legend()
        pyplot.title(backend_class.__name__)
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        for datum_x in data_x.values():
            np.testing.assert_array_equal(data_x["initial"], datum_x)

        peaks_expected = {
            "initial": (30, 0.017),
            "n_f = 1": (1600, 0.015),
            "n_f = 4": (500, 0.01),
            "n_f = 16": (200, 0.0075),
            "n_f = 64": (24, 0.006),
        }

        for lbl, x_y in peaks_expected.items():
            print(lbl)
            peak = np.argmax(data_y[lbl][0])
            np.testing.assert_approx_equal(data_x[lbl][peak], x_y[0], significant=1)
            np.testing.assert_approx_equal(
                data_y[lbl][0][peak] * settings.rho, x_y[1], significant=1
            )

    @staticmethod
    def test_fig_4b(backend_class, plot=False):
        # arrange
        settings0 = Settings0D()
        settings0.n_sd = N_SD
        settings0.radius_bins_edges = bins_edges(64)
        x_0 = Settings0D.X0
        mu_vals = [4 * x_0, x_0, x_0 / 4]
        data_x = {}
        data_y = {}

        # act
        lbl = "initial"
        (data_x[lbl], data_y[lbl], _) = run_box_breakup(settings0, [0], backend_class)
        for i, mu_val in enumerate(mu_vals):
            settings = Settings0D(
                fragmentation=Gaussian(
                    mu=mu_val, sigma=mu_val / 2, vmin=(1 * si.um) ** 3, nfmax=None
                ),
                warn_overflows=False,
                seed=44,
            )
            settings.dt = DT
            settings.n_sd = settings0.n_sd
            settings.radius_bins_edges = settings0.radius_bins_edges
            settings.coal_eff = ConstEc(Ec=0.99)
            lbl = r"$\mu$ = " + str(round(mu_val / x_0, 2)) + "X$_0$"
            (data_x[lbl], data_y[lbl], _) = run_box_breakup(
                settings, [120], backend_class
            )

        # plot
        pyplot.step(
            data_x["initial"],
            data_y["initial"][0] * settings.rho,
            color="k",
            linestyle="--",
            label=lbl,
        )
        for i, mu_val in enumerate(mu_vals):
            lbl = r"$\mu$ = " + str(round(mu_val / x_0, 2)) + "X$_0$"
            pyplot.step(
                data_x[lbl],
                data_y[lbl][0] * settings.rho,
                color=CMAP(i / len(mu_vals)),
                linestyle="-",
                label=lbl
                if lbl not in pyplot.gca().get_legend_handles_labels()[1]
                else "",
            )

        pyplot.xscale("log")
        pyplot.xlabel("particle radius (um)")
        pyplot.ylabel("dm/dlnr (kg/m$^3$ / unit(ln R)")
        pyplot.legend()
        pyplot.title(backend_class.__name__)
        pyplot.tight_layout()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        for datum_x in data_x.values():
            np.testing.assert_array_equal(data_x["initial"], datum_x)

        peaks_expected = {
            "initial": (30, 0.017),
            r"$\mu$ = 0.25X$_0$": (25.5, 0.015),
            r"$\mu$ = 4.0X$_0$": (62, 0.010),
            r"$\mu$ = 1.0X$_0$": (42, 0.019),
        }

        for lbl, x_y in peaks_expected.items():
            print(lbl)
            peak = np.argmax(data_y[lbl][0])
            np.testing.assert_approx_equal(data_x[lbl][peak], x_y[0], significant=1)
            np.testing.assert_approx_equal(
                data_y[lbl][0][peak] * settings.rho, x_y[1], significant=1
            )
