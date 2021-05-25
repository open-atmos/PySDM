from PySDM.products.product import MoistEnvironmentProduct


class RelativeHumidity(MoistEnvironmentProduct):

    def __init__(self):
        super().__init__(
            description="Relative humidity",
            name="RH",
            unit="%"
        )

    def get(self):
        super().get()
        self.buffer *= 100
        return self.buffer
