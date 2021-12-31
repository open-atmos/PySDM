# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect
import numpy as np
from matplotlib import pyplot
from PySDM import Formulae
from PySDM.physics import constants_defaults as const


def test_saturation_vapour_pressures(plot=False):
    # Arrange
    formulae = {
        k: Formulae(saturation_vapour_pressure=k)
        for k in ('FlatauWalkoCotton', 'AugustRocheMagnus')
    }
    temperature = np.linspace(-.2, .4)

    # Plot
    pyplot.axhline(const.p_tri, label='triple point', color='red')
    pyplot.axvline(const.T_tri - const.T0, color='red')
    for key, val in formulae.items():
        for name, func in inspect.getmembers(val.saturation_vapour_pressure):
            if name[:2] not in ('__', 'a_'):
                pyplot.plot(temperature, func(temperature), label=f"{key}::{name}")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel('T [C]')
    pyplot.ylabel('p [Pa]')
    if plot:
        pyplot.show()

    # Assert
    temperature = np.linspace(-20, 20, 100)
    np.testing.assert_allclose(
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton')
            .saturation_vapour_pressure.pvs_Celsius(temperature),
        Formulae(saturation_vapour_pressure='AugustRocheMagnus')
            .saturation_vapour_pressure.pvs_Celsius(temperature),
        rtol=1e-2
    )
    temperature = np.linspace(-20, 0.3, 100)
    np.testing.assert_array_less(
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton')
            .saturation_vapour_pressure.ice_Celsius(temperature),
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton')
            .saturation_vapour_pressure.pvs_Celsius(temperature)
    )
    temperature = np.linspace(0.35, 20, 100)
    np.testing.assert_array_less(
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton')
            .saturation_vapour_pressure.pvs_Celsius(temperature),
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton')
            .saturation_vapour_pressure.ice_Celsius(temperature)
    )
