"""common code for attributes offering an option to neglect temperature variation,
intended for use with Parcel environment only"""

from PySDM.environments import Parcel


class TemperatureVariationOptionAttribute:  # pylint: disable=too-few-public-methods
    """base class"""

    def __init__(self, builder, neglect_temperature_variations: bool):
        assert isinstance(builder.particulator.environment, Parcel)
        self.neglect_temperature_variations = neglect_temperature_variations
        self.initial_temperature = builder.particulator.Storage.from_ndarray(
            builder.particulator.environment["T"].to_ndarray()
        )
