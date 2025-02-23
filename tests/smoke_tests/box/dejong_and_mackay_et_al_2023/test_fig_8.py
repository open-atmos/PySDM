# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.deJong_Mackay_et_al_2023 import Settings0D, run_box_breakup

from PySDM.backends import CPU, GPU
from PySDM.dynamics.collisions.breakup_fragmentations import Straub2010Nf
from PySDM.dynamics.collisions.coalescence_efficiencies import Straub2010Ec
from PySDM.physics import si


@pytest.mark.parametrize(
    "backend_class",
    (CPU, pytest.param(GPU, marks=pytest.mark.xfail(strict=True))),  # TODO #987
)
def test_fig_8(backend_class, plot=False):
    # arrange
    settings = Settings0D(
        fragmentation=Straub2010Nf(vmin=Settings0D.X0 * 1e-3, nfmax=10),
        seed=44,
        warn_overflows=False,
    )
    steps = [0, 1200, 3600]
    settings._steps = steps  # pylint: disable=protected-access
    settings.n_sd = 2**11
    settings.radius_bins_edges = np.logspace(
        np.log10(4 * si.um), np.log10(5e3 * si.um), num=64, endpoint=True
    )
    settings.coal_eff = Straub2010Ec()

    # act
    res = run_box_breakup(settings, backend_class=backend_class)
    data_x, data_y = res.x, res.y

    # plot
    cmap = matplotlib.colormaps["viridis"]
    for j, step in enumerate(steps):
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
    pyplot.title(backend_class.__name__)
    pyplot.legend()
    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # assert
    peaks_expected = {
        0: (34, 0.019),
        1200: (2839, 0.03),
        3600: (2839, 0.02),
    }

    for j, step in enumerate(steps):
        peak = np.argmax(data_y[j])
        np.testing.assert_approx_equal(
            actual=data_x[peak], desired=peaks_expected[step][0], significant=1
        )


# TODO #1048
#         np.testing.assert_approx_equal(
#             actual=data_y[j][peak] * settings.rho,
#             desired=peaks_expected[step][1],
#             significant=2,
#         )
