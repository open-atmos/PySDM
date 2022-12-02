# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import matplotlib
import numpy as np
from matplotlib import pyplot
from PySDM_examples.deJong_Mackay_2022 import Settings0D, run_box_breakup

from PySDM.dynamics.collisions.breakup_fragmentations import Straub2010Nf
from PySDM.dynamics.collisions.coalescence_efficiencies import Straub2010Ec
from PySDM.physics import si


def test_fig_5(plot=False):
    # arrange
    settings = Settings0D(Straub2010Nf(vmin=Settings0D.X0 * 1e-3, nfmax=10), seed=44)
    steps = [0, 30, 60, 180, 540]
    settings._steps = steps  # pylint: disable=protected-access
    settings.n_sd = 2**11
    settings.warn_overflows = False
    settings.radius_bins_edges = np.logspace(
        np.log10(10 * si.um), np.log10(2e3 * si.um), num=32, endpoint=True
    )
    settings.coal_eff = Straub2010Ec()

    # act
    (data_x, data_y, _) = run_box_breakup(settings)

    # plot
    cmap = matplotlib.cm.get_cmap("viridis")
    for (j, step) in enumerate(steps):
        if j == 0:
            kwargs = {"color": "k", "linestyle": "--", "label": "initial"}
        else:
            kwargs = {
                "color": cmap(j / len(steps)),
                "linestyle": "-",
                "label": f"t = {step}s",
            }
        pyplot.step(data_x, data_y[j] * settings.rho, **kwargs)
    pyplot.xscale("log")
    pyplot.xlabel("particle radius (um)")
    pyplot.ylabel("dm/dlnr (kg/m$^3$ / unit(ln R)")
    pyplot.legend()
    if plot:
        pyplot.show()

    # assert
    peaks_expected = {
        0: (33, 0.018),
        30: (92, 0.011),
        60: (305, 0.012),
        180: (717, 0.015),
        540: (717, 0.015),
    }

    for (j, step) in enumerate(steps):
        print(step)
        peak = np.argmax(data_y[j])
        np.testing.assert_approx_equal(
            actual=data_x[peak], desired=peaks_expected[step][0], significant=2
        )
        np.testing.assert_approx_equal(
            actual=data_y[j][peak] * settings.rho,
            desired=peaks_expected[step][1],
            significant=2,
        )
