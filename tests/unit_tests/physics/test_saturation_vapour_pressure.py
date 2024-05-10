# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect
import pytest

import numpy as np
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.formulae import _choices
from PySDM.physics import constants_defaults as const
from PySDM.physics import saturation_vapour_pressure, si


class TestSaturationVapourPressure:
    @staticmethod
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
                    if not (
                        key in ("AugustRocheMagnus", "Wexler1976", "Bolton1980")
                        and name == "ice_Celsius"
                    ):
                        pyplot.plot(
                            temperature, func(temperature), label=f"{key}::{name}"
                        )
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
            if not choice in ("AugustRocheMagnus", "Bolton1980", "Wexler1976"):
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

    @staticmethod
    @pytest.mark.parametrize(
        "T_C, expected_es_mb",
        (
            (-40, 0.1905),
            (-30, 0.5106),
            (-20, 1.2563),
            (-10, 2.8657),
            (0, 6.1121),
            (10, 12.279),
            (20, 23.385),
            (30, 42.452),
            (40, 73.813),
        ),
    )
    def test_wexler_1976_table_1(T_C, expected_es_mb):
        formulae = Formulae(saturation_vapour_pressure="Wexler1976")
        actual_es = formulae.saturation_vapour_pressure.pvs_Celsius(T_C)
        np.testing.assert_approx_equal(
            actual=actual_es, desired=expected_es_mb * si.mbar, significant=4
        )
