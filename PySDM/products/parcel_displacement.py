from PySDM.environments import Parcel
from PySDM.impl.product import Product


class ParcelDisplacement(Product):

    def __init__(self):
        super().__init__(
            description="Parcel displacement",
            name="z",
            unit="m"
        )
        self.environment = None

    def register(self, builder):
        super().register(builder)
        assert isinstance(builder.particulator.environment, Parcel)
        self.environment = builder.particulator.environment

    def get(self):
        self.download_to_buffer(self.environment['z'])
        return self.buffer
