# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

import numpy as np
from matplotlib import pyplot
from PySDM_examples.Bieli_et_al_2022.settings import Settings
from PySDM_examples.Bieli_et_al_2022.simulation import make_core

from PySDM import Formulae
from PySDM.physics import si


def test_moments(plot=False):
    # arrange
    settings = Settings(Formulae(fragmentation_function="Feingold1988"))
    if "CI" in os.environ:
        settings.n_sd = 10
    else:
        settings.n_sd = 100

    moments = {
        i: np.zeros((3, len(settings.output_steps)))
        for i, _ in enumerate(settings.coal_effs)
    }

    # act
    for i, coal_eff in enumerate(settings.coal_effs):
        particulator = make_core(settings, coal_eff)

        j = 0
        for step in settings.output_steps:
            particulator.run(step - particulator.n_steps)
            moments[i][:, j] = [
                particulator.products["M0"].get()[0],
                particulator.products["M1"].get()[0],
                particulator.products["M2"].get()[0],
            ]
            j += 1
        moments[i][1, :] *= settings.rho / si.g
        moments[i][2, :] *= settings.rho**2 / si.g**2
        moments[i] *= 1 / settings.dv * si.cm**3

    # plot
    __, axs = pyplot.subplots(nrows=1, ncols=3, figsize=(8, 3))
    for i, _ in enumerate(settings.coal_effs):
        for moment in range(3):
            axs[moment].plot(settings.output_steps, moments[i][moment, :])

    axs[0].set_xlabel("time (s)")
    axs[1].set_xlabel("time (s)")
    axs[2].set_xlabel("time (s)")
    axs[0].set_ylabel("$M_0$ (1/cm$^3$)")
    axs[1].set_ylabel("$M_1$ (g/cm$^3$)")
    axs[2].set_ylabel("$M_2$ (g$^2$/cm$^3$)")

    axs[0].set_ylim([0, 2e4])
    axs[1].set_ylim([0, 5e-6])
    axs[2].set_ylim([0, 2e-14])
    pyplot.legend(["E_c=0.8", "E_c=0.9", "E_c=1.0"])
    pyplot.tight_layout()

    if plot:
        pyplot.show()

    # assert on mass conservation
    tol = 1e-10
    for i, _ in enumerate(settings.coal_effs):
        assert moments[i][1, -1] <= moments[i][1, 0] * (1 + tol)
        assert moments[i][1, -1] >= moments[i][1, 0] * (1 - tol)
