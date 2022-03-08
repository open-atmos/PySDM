# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

from matplotlib import pyplot
from PySDM_examples.deJong_Mackay_2022.settings import Settings
from PySDM_examples.deJong_Mackay_2022.simulation import make_core

from PySDM.physics import si


def test_collision(plot=False):
    settings = Settings()
    if "CI" in os.environ:
        settings.n_sd = 10
    else:
        settings.n_sd = 100

    particulator = make_core(settings)

    for step in settings.output_steps:
        particulator.run(step - particulator.n_steps)
        if plot:
            pyplot.step(
                x=settings.radius_bins_edges[:-1] / si.micrometres,
                y=particulator.products["dv/dlnr"].get() * settings.rho,
                where="post",
                label="t = {step*settings.dt}s",
            )
    if plot:
        pyplot.xscale("log")
        pyplot.xlabel("radius (um)")
        pyplot.ylabel("dm/dlnr")
        pyplot.legend([0, 1, 2])

    # TODO #744: add asserts here to check whether stuff is correct
    assert True
