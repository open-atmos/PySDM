from PySDM.impl.product import MoistEnvironmentProduct


class DryAirPotentialTemperature(MoistEnvironmentProduct):

    def __init__(self):
        super().__init__(
            description="Dry-air potential temperature",
            name="thd",
            unit="K"
        )
