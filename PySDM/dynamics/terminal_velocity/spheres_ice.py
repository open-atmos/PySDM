"""
terminal velocity of solid ice spheres
"""


class IceSphere:  # pylint: disable=too-few-public-methods,too-many-arguments
    def __init__(self, particulator):
        self.particulator = particulator

    def __call__(self, output, signed_water_mass, cell_id, temperature, pressure):
        self.particulator.backend.terminal_velocity_ice_spheres(
            values=output.data,
            signed_water_mass=signed_water_mass.data,
            cell_id=cell_id.data,
            temperature=temperature.data,
            pressure=pressure.data,
        )
