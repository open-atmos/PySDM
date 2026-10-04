"""common code for attributes offering an option to neglect temperature variation,
intended for use with Parcel environment only"""


class TemperatureVariationOptionAttribute:  # pylint: disable=too-few-public-methods
    """base class"""

    def __init__(self, particulator, neglect_temperature_variations: bool):
        if neglect_temperature_variations:
            assert particulator.environment.mesh.dimension == 0
        self.neglect_temperature_variations = neglect_temperature_variations
        self.initial_temperature = (
            particulator.Storage.from_ndarray(
                particulator.environment["T"].to_ndarray()
            )
            if neglect_temperature_variations
            else None
        )
