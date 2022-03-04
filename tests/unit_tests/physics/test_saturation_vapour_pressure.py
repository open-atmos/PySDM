# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect

import numpy as np
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.formulae import _choices
from PySDM.physics import constants_defaults as const
from PySDM.physics import saturation_vapour_pressure


def test_saturation_vapour_pressure(plot=False):
    # Arrange
    choices = _choices(saturation_vapour_pressure)
    formulae = {k: Formulae(saturation_vapour_pressure=k) for k in choices}
    temperature = np.linspace(-0.2, 0.4)

    # Plot
    pyplot.axhline(const.p_tri, label="triple point", color="red")
    pyplot.axvline(const.T_tri - const.T0, color="red")
    for key, val in formulae.items():
        for name, func in inspect.getmembers(val.saturation_vapour_pressure):
            if name[:2] not in ("__", "a_"):
                if not (key == "AugustRocheMagnus" and name == "ice_Celsius"):
                    pyplot.plot(temperature, func(temperature), label=f"{key}::{name}")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("T [C]")
    pyplot.ylabel("p [Pa]")
    if plot:
        pyplot.show()

    # Assert
    temperature = np.linspace(-20, 20, 100)
    choices_keys = tuple(choices.keys())
    for choice in choices_keys[1:]:
        np.testing.assert_allclose(
            Formulae(
                saturation_vapour_pressure=choices_keys[0]
            ).saturation_vapour_pressure.pvs_Celsius(temperature),
            Formulae(
                saturation_vapour_pressure=choice
            ).saturation_vapour_pressure.pvs_Celsius(temperature),
            rtol=2e-2,
        )

    for choice in choices_keys[1:]:
        if choice != "AugustRocheMagnus":
            temperature = np.linspace(-20, 0, 100)
            np.testing.assert_array_less(
                Formulae(
                    saturation_vapour_pressure="FlatauWalkoCotton"
                ).saturation_vapour_pressure.ice_Celsius(temperature),
                Formulae(
                    saturation_vapour_pressure=choice
                ).saturation_vapour_pressure.pvs_Celsius(temperature),
            )
            temperature = np.linspace(1, 1, 100)
            np.testing.assert_array_less(
                Formulae(
                    saturation_vapour_pressure="FlatauWalkoCotton"
                ).saturation_vapour_pressure.pvs_Celsius(temperature),
                Formulae(
                    saturation_vapour_pressure=choice
                ).saturation_vapour_pressure.ice_Celsius(temperature),
            )
