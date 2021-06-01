from PySDM.products.product import MoistEnvironmentProduct


class Temperature(MoistEnvironmentProduct):

    def __init__(self):
        super().__init__(
            description="Temperature",
            name="T",
            unit="K"
        )
