# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect

import numpy as np
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.physics import constants_defaults as const


def test_thermal_conductivities(plot=False):
    # Arrange
    formulae = {k: Formulae(diffusion_thermics=k) for k in ("LoweEtAl2019",)}
    temperature = np.linspace(-20.2, 20.4) + const.T_tri
    pressure = const.p_tri

    # Plot
    pyplot.axhline(const.K0, label="TracyWelchPorter", color="red")
    for key, val in formulae.items():
        for name, func in inspect.getmembers(val.diffusion_thermics):
            if name[:2] not in ("__", "a_", "D"):
                pyplot.plot(
                    temperature, func(temperature, pressure), label=f"{key}::{name}"
                )
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("T [C]")
    pyplot.ylabel("k [J/m/s/K]")
    if plot:
        pyplot.show()

    # Assert
    temperature = np.linspace(-20, 20, 100) + const.T_tri
    np.testing.assert_allclose(
        Formulae(diffusion_thermics="TracyWelchPorter").diffusion_thermics.K(
            temperature, pressure
        ),
        Formulae(diffusion_thermics="LoweEtAl2019").diffusion_thermics.K(
            temperature, pressure
        ),
        rtol=1e-1,
    )
