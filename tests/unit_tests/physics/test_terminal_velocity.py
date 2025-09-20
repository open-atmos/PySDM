"""tests for terminal velocity formulae (both liquid and ice phase)"""

from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.physics import constants_defaults, terminal_velocity, terminal_velocity_ice


class TestTerminalVelocity:
    @staticmethod
    def test_unit_liquid_rogers_yau():
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            radius = 44 * si.um

            # act
            velocity = terminal_velocity.RogersYau.v_term(
                constants_defaults, radius=radius
            )

            # assert
            assert velocity.check("[length] / [time]")  # pylint: disable=no-member

    @staticmethod
    def test_unit_solid_spherical():
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            radius = 44 * si.um
            dynamic_viscocity = 0.666 * si.mPa * si.s

            # act
            prefactor = terminal_velocity_ice.IceSphere.stokes_regime(
                constants_defaults, radius=radius, dynamic_viscosity=dynamic_viscocity
            )
            velocity = terminal_velocity_ice.IceSphere.v_base_term(
                constants_defaults, radius=radius, prefactor=prefactor
            )

            # assert
            assert velocity.check("[length] / [time]")

    @staticmethod
    def test_unit_solid_columnar():
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            mass = 44 * si.ug
            temperature = 250 * si.K
            pressure = 1000 * si.hPa

            # act
            correction_factor = (
                terminal_velocity_ice.ColumnarIceCrystal.atmospheric_correction_factor(
                    constants_defaults, temperature=temperature, pressure=pressure
                )
            )
            velocity = (
                correction_factor
                * terminal_velocity_ice.ColumnarIceCrystal.v_base_term(
                    constants_defaults, mass=mass
                )
            )

            # assert
            assert velocity.check("[length] / [time]")
