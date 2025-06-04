"""
Tests checking if, for the example case from Fig 4 in
[Shima et al. 2009](https://doi.org/10.1002/qj.441),
the simulations converge towards analytic solution.
"""

from inspect import signature
from itertools import islice

import pytest
from matplotlib import pyplot
from PySDM_examples.Shima_et_al_2009.example import run
from PySDM_examples.Shima_et_al_2009.settings import Settings
from PySDM_examples.Shima_et_al_2009.spectrum_plotter import SpectrumPlotter

from PySDM.physics import si

COLORS = ("red", "green", "blue")


class TestConvergence:  # pylint: disable=missing-class-docstring
    @staticmethod
    @pytest.mark.parametrize(
        "adaptive, dt",
        (
            pytest.param(False, 100 * si.s, marks=pytest.mark.xfail(strict=True)),
            (True, 100 * si.s),
            pytest.param(False, 50 * si.s, marks=pytest.mark.xfail(strict=True)),
            (True, 50 * si.s),
        ),
    )
    def test_convergence_with_sd_count(dt, adaptive, plot=False):
        """check if increasing the number of super particles indeed
        reduces the error of the simulation (vs. analytic solution)"""
        # arrange
        settings = Settings(steps=[3600])
        settings.adaptive = adaptive
        plotter = SpectrumPlotter(settings)
        errors = {}

        # act
        for i, ln2_nsd in enumerate((11, 15, 19)):
            settings.dt = dt
            settings.n_sd = 2**ln2_nsd
            values, _ = run(settings)

            title = (
                ""
                if i != 0
                else (
                    f"{settings.dt=}  settings.times={settings.steps}  {settings.adaptive=}"
                )
            )
            errors[ln2_nsd] = plotter.plot(
                **dict(
                    islice(
                        {  # supporting older versions of PySDM-examples
                            "t": settings.steps[-1],
                            "spectrum": values[tuple(values.keys())[-1]],
                            "label": f"{ln2_nsd=}",
                            "color": COLORS[i],
                            "title": title,
                            "add_error_to_label": True,
                        }.items(),
                        len(signature(plotter.plot).parameters),
                    )
                )
            )

        # plot
        if plot:
            plotter.show()
        else:
            # https://github.com/matplotlib/matplotlib/issues/9970
            pyplot.xscale("linear")
            pyplot.clf()

        # assert monotonicity (i.e., the larger the sd count, the smaller the error)
        assert tuple(errors.keys()) == tuple(sorted(errors.keys()))
        assert tuple(errors.values()) == tuple(reversed(sorted(errors.values())))

    @staticmethod
    def test_convergence_with_timestep():
        """ditto for timestep"""
        pytest.skip("# TODO #1189")
