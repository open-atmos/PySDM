"""
[Spichtinger & Gierens 2009](https://doi.org/10.5194/acp-9-685-2009)
Eq. (18) .Assumed shape is columnar based on empirical parameterizations of
[Heymsfield & Iaquinta (2000)](https://doi.org/10.1175/1520-0469(2000)057%3C0916:CCTV%3E2.0.CO;2)
[Barthazy & Schefold (2006)](https://doi.org/10.1016/j.atmosres.2005.12.009)
"""


class ColumnarIceCrystal:  # pylint: disable=too-few-public-methods,too-many-arguments
    def __init__(self, particulator):
        self.particulator = particulator

    def __call__(self, output, signed_water_mass, cell_id, temperature, pressure):
        self.particulator.backend.terminal_velocity_columnar_ice_crystals(
            values=output.data,
            signed_water_mass=signed_water_mass.data,
            cell_id=cell_id.data,
            temperature=temperature.data,
            pressure=pressure.data,
        )
