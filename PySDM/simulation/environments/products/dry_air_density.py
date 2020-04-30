from ...product import Product

class DryAirDensity(Product):
    def __init__(self, environment):
        self.environment = environment
        super().__init__(particles=environment.particles,
                         description="Dry-air density",
                         name="rhod",
                         unit="kg/m^3",
                         range=(0.95, 1.3),
                         scale="linear",
                         shape=environment.particles.mesh.grid)

    def get(self):
        self.download_to_buffer(self.environment['rhod'])
        return self.buffer

