"""
[Spichtinger & Gierens 2009](https://doi.org/10.5194/acp-9-685-2009)
Eq. (18) .Assumed shape is columnar based on empirical parameterizations of
[Heymsfield & Iaquinta (2000)](https://doi.org/10.1175/1520-0469(2000)057<0916:CCTV>2.0.CO;2)
[Barthazy & Schefold (2006)](https://doi.org/10.1016/j.atmosres.2005.12.009)
"""

class ColumnarIceCrystal:  # pylint: disable=too-few-public-methods
    def __init__(self, particulator):
        self.particulator = particulator

    def __call__(self, output, mass):
        self.particulator.backend.terminal_velocity(
            values=output.data,
            mass=mass.data,
        )