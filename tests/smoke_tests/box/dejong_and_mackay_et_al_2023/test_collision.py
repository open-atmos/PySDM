# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

from matplotlib import pyplot
from PySDM_examples.deJong_Mackay_et_al_2023 import (
    Settings0D,
    run_box_breakup,
    run_box_NObreakup,
)


def test_collision(backend_class, plot=False):
    settings = Settings0D(warn_overflows=False)
    t_steps = [0, 100, 200]
    if "CI" in os.environ:
        settings.n_sd = 10
    else:
        settings.n_sd = 100

    (x1, y1, __) = run_box_NObreakup(settings, t_steps, backend_class)
    res2 = run_box_breakup(settings, t_steps, backend_class)
    x2, y2 = res2.x, res2.y

    for step in settings.output_steps:
        pyplot.step(
            x=x1,
            y=y1[step],
            where="post",
            label=f"NO breakup, t = {step*settings.dt}s",
        )
        pyplot.step(
            x=x2,
            y=y2[step],
            where="post",
            label=f"WITH breakup, t = {step*settings.dt}s",
        )
    pyplot.xscale("log")
    pyplot.xlabel("radius (um)")
    pyplot.ylabel("dm/dlnr")
    pyplot.legend([0, 1, 2])
    pyplot.title(backend_class.__name__)

    if plot:
        pyplot.show()
    else:
        pyplot.clf()

    # TODO #744: add asserts here to check whether stuff is correct
    assert True
