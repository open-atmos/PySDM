# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

from matplotlib import pyplot
from PySDM_examples.deJong_Mackay_2022 import (
    Settings0D,
    run_box_breakup,
    run_box_NObreakup,
)


def test_collision(plot=False):
    settings = Settings0D()
    t_steps = [0, 100, 200]
    if "CI" in os.environ:
        settings.n_sd = 10
    else:
        settings.n_sd = 100

    (x1, y1, __) = run_box_NObreakup(settings, t_steps)
    (x2, y2, __) = run_box_breakup(settings, t_steps)

    for step in settings.output_steps:
        if plot:
            pyplot.step(
                x=x1,
                y=y1,
                where="post",
                label=f"NO breakup, t = {step*settings.dt}s",
            )
            pyplot.step(
                x=x2,
                y=y2,
                where="post",
                label=f"WITH breakup, t = {step*settings.dt}s",
            )
    if plot:
        pyplot.xscale("log")
        pyplot.xlabel("radius (um)")
        pyplot.ylabel("dm/dlnr")
        pyplot.legend([0, 1, 2])

    # TODO #744: add asserts here to check whether stuff is correct
    assert True
