"""common code for attributes offering an option to neglect temperature variation,
intended for use with Parcel environment only"""


class TemperatureVariationOptionAttribute:  # pylint: disable=too-few-public-methods
    """base class"""

    def __init__(self, builder, neglect_temperature_variations: bool):
        if neglect_temperature_variations:
            assert builder.particulator.environment.mesh.dimension == 0
        self.neglect_temperature_variations = neglect_temperature_variations
        self.initial_temperature = (
            builder.particulator.Storage.from_ndarray(
                builder.particulator.environment["T"].to_ndarray()
            )
            if neglect_temperature_variations
            else None
        )
