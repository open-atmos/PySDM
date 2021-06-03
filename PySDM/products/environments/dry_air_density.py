from PySDM.products.product import MoistEnvironmentProduct


class DryAirDensity(MoistEnvironmentProduct):

    def __init__(self):
        super().__init__(
            description="Dry-air density",
            name="rhod",
            unit="kg/m^3"
        )

