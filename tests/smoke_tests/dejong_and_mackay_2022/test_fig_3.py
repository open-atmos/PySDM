# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import matplotlib
import numpy as np
from matplotlib import pyplot
from PySDM_examples.deJong_Mackay_2022 import Settings0D, run_box_breakup

from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc, Straub2010Ec
from PySDM.physics.constants import si

R_MIN = 0.1 * si.um
V_MIN = 4 / 3 * np.pi * R_MIN**3
EC_VALS = [1.0, 0.95, 0.9, 0.8]
BINS = 32
N_SD = 2**10


def test_fig_3_reduced_resolution(plot=False):
    # arrange
    settings = Settings0D(fragmentation=AlwaysN(n=8, vmin=V_MIN), seed=44)
    settings.n_sd = N_SD
    settings.radius_bins_edges = np.logspace(
        np.log10(10 * si.um), np.log10(10000 * si.um), num=BINS, endpoint=True
    )
    settings.warn_overflows = False
    settings._steps = [200]  # pylint: disable=protected-access
    data_x = {}
    data_y = {}

    # act
    lbl = "initial"
    (data_x[lbl], data_y[lbl], _) = run_box_breakup(settings, [0])

    for (i, ec_value) in enumerate(EC_VALS):
        settings.coal_eff = ConstEc(Ec=ec_value)
        lbl = "Ec = " + str(ec_value)
        if ec_value == 1.0:
            lbl = "Ec = 1.0"
        (data_x[lbl], data_y[lbl], _) = run_box_breakup(settings)

    lbl = "Straub 2010"
    settings.coal_eff = Straub2010Ec()
    (data_x[lbl], data_y[lbl], _) = run_box_breakup(settings)

    # plot
    lbl = "initial"
    pyplot.step(
        data_x[lbl], data_y[lbl][0] * settings.rho, color="k", linestyle="--", label=lbl
    )
    for (i, ec_value) in enumerate(EC_VALS):
        lbl = "Ec = " + str(ec_value)
        if ec_value == 1.0:
            lbl = "Ec = 1.0"
        pyplot.step(
            data_x[lbl],
            data_y[lbl][0] * settings.rho,
            color=matplotlib.cm.get_cmap("viridis")(i / len(EC_VALS)),
            label=lbl if lbl not in pyplot.gca().get_legend_handles_labels()[1] else "",
        )

    lbl = "Straub 2010"
    pyplot.step(
        data_x[lbl],
        data_y[lbl][0] * settings.rho,
        color="m",
        label=lbl if lbl not in pyplot.gca().get_legend_handles_labels()[1] else "",
    )

    pyplot.xscale("log")
    pyplot.xlabel("particle radius (um)")
    pyplot.ylabel("dm/dlnR (kg/m$^3$ / unit(ln R)")
    pyplot.legend()
    if plot:
        pyplot.show()

    # assert
    for datum_x in data_x.values():
        np.testing.assert_array_equal(data_x["initial"], datum_x)

    peaks_expected = {
        "initial": (30, 0.017),
        "Ec = 1.0": (1600, 0.015),
        "Ec = 0.95": (800, 0.01),
        "Ec = 0.9": (200, 0.01),
        "Ec = 0.8": (20, 0.0125),
        "Straub 2010": (200, 0.0125),
    }

    for lbl, x_y in peaks_expected.items():
        print(lbl)
        peak = np.argmax(data_y[lbl][0])
        np.testing.assert_approx_equal(data_x[lbl][peak], x_y[0], significant=1)
        np.testing.assert_approx_equal(
            data_y[lbl][0][peak] * settings.rho, x_y[1], significant=1
        )
