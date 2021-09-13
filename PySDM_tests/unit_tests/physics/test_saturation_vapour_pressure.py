import numpy as np
from matplotlib import pylab
from PySDM.physics import Formulae, constants as const
import inspect

def test_saturation_vapour_pressures(plot=False):
    # Arrange
    formulae = {
        k: Formulae(saturation_vapour_pressure=k)
        for k in ('FlatauWalkoCotton', 'AugustRocheMagnus')
    }
    T = np.linspace(-.2, .4)

    # Plot
    if plot:
        pylab.axhline(const.p_tri, label='triple point', color='red')
        pylab.axvline(const.T_tri - const.T0, color='red')
        for k, v in formulae.items():
            for name, func in inspect.getmembers(v.saturation_vapour_pressure):
                if not name.startswith('__'):
                    pylab.plot(T, func(T), label=f"{k}::{name}")
        pylab.grid()
        pylab.legend()
        pylab.xlabel('T [C]')
        pylab.ylabel('p [Pa]')
        pylab.show()

    # Assert
    T = np.linspace(-20, 20, 100)
    np.testing.assert_allclose(
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton').saturation_vapour_pressure.pvs_Celsius(T),
        Formulae(saturation_vapour_pressure='AugustRocheMagnus').saturation_vapour_pressure.pvs_Celsius(T),
        rtol=1e-2
    )
    T = np.linspace(-20, 0.3, 100)
    np.testing.assert_array_less(
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton').saturation_vapour_pressure.ice_Celsius(T),
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton').saturation_vapour_pressure.pvs_Celsius(T)
    )
    T = np.linspace(0.35, 20, 100)
    np.testing.assert_array_less(
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton').saturation_vapour_pressure.pvs_Celsius(T),
        Formulae(saturation_vapour_pressure='FlatauWalkoCotton').saturation_vapour_pressure.ice_Celsius(T)
    )