# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect

import numpy as np
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.physics import constants_defaults as const


def test_latent_heats(plot=False):
    # Arrange
    formulae = {k: Formulae(latent_heat=k) for k in ("Kirchhoff", "Lowe2019")}
    temperature = np.linspace(-20, 20) + const.T_tri

    # Plot
    pyplot.axhline(const.l_tri, label="triple point", color="red")
    pyplot.axvline(const.T_tri, color="red")
    for key, val in formulae.items():
        for name, func in inspect.getmembers(val.latent_heat):
            if name[:2] not in ("__", "a_"):
                pyplot.plot(temperature, func(temperature), label=f"{key}::{name}")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("T [K]")
    pyplot.ylabel("Lv [J/kg]")
    if plot:
        pyplot.show()

    # Assert
    temperature = np.linspace(-20, 20, 100) + const.T_tri
    np.testing.assert_allclose(
        Formulae(latent_heat="Kirchhoff").latent_heat.lv(temperature),
        Formulae(latent_heat="Lowe2019").latent_heat.lv(temperature),
        rtol=1e-2,
    )
