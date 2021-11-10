from PySDM.impl.product import MoistEnvironmentProduct


class Pressure(MoistEnvironmentProduct):

    def __init__(self):
        super().__init__(
            description="Pressure",
            name="p",
            unit="Pa"
        )
