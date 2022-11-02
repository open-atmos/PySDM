# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import matplotlib
import numpy as np
from matplotlib import pyplot
from PySDM_examples.deJong_Mackay_2022 import Settings0D, run_box_breakup

from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc, Straub2010Ec
from PySDM.physics.constants import si


def test_fig_3(plot=True):
    # TODO #744
    # pylint: disable=protected-access

    # arrange
    settings = Settings0D()

    rmin = 0.1 * si.um
    vmin = 4 / 3 * np.pi * rmin**3
    settings.fragmentation = AlwaysN(n=8, vmin=vmin)

    settings.n_sd = 2**13
    settings.radius_bins_edges = np.logspace(
        np.log10(1.0 * si.um), np.log10(10000 * si.um), num=128, endpoint=True
    )

    settings.warn_overflows = False
    settings._steps = [0, 100, 200]
    cmap = matplotlib.cm.get_cmap("viridis")

    ec_vals = [1.0, 0.95, 0.9, 0.8]

    # act / plot
    (data_x, data_y, _) = run_box_breakup(settings, [0])
    _, axs = pyplot.subplots(ncols=2, sharey=True, figsize=(10, 4), dpi=200)
    axs[0].step(
        data_x, data_y[0] * settings.rho, color="k", linestyle="--", label="initial"
    )
    axs[1].step(
        data_x, data_y[0] * settings.rho, color="k", linestyle="--", label="initial"
    )
    for (i, ec_value) in enumerate(ec_vals):
        settings.coal_eff = ConstEc(Ec=ec_value)
        (data_x, data_y, _) = run_box_breakup(settings)
        for (j, _) in enumerate(settings._steps):
            if j == 0:
                continue
            lbl = "Ec = " + str(ec_value)
            if ec_value == 1.0:
                lbl = "Ec = 1.0"
            axs[j - 1].step(
                data_x,
                data_y[j] * settings.rho,
                color=cmap(i / len(ec_vals)),
                label=lbl
                if lbl not in pyplot.gca().get_legend_handles_labels()[1]
                else "",
            )

    settings.coal_eff = Straub2010Ec()
    (data_x, data_y, _) = run_box_breakup(settings)
    for (j, _) in enumerate(settings._steps):
        if j == 0:
            continue
        lbl = "Straub 2010"
        axs[j - 1].step(
            data_x,
            data_y[j] * settings.rho,
            color="m",
            label=lbl if lbl not in pyplot.gca().get_legend_handles_labels()[1] else "",
        )

    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[0].set_xlabel("particle radius (um)")
    axs[1].set_xlabel("particle radius (um)")
    axs[0].set_ylabel("dm/dlnR (kg/m$^3$ / unit(ln R)")
    axs[0].legend()
    axs[0].set_title("t = 100sec")
    axs[1].set_title("t = 200sec")
    if plot:
        pyplot.show()

    # assert
    # TODO #744
