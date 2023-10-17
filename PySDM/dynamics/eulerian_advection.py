"""
wrapper class for triggering integration in the Eulerian advection solver
"""


class EulerianAdvection:
    def __init__(self, solvers):
        self.solvers = solvers
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

    def __call__(self):
        self.particulator.environment.get_predicted(
            "water_vapour_mixing_ratio"
        ).download(
            self.particulator.environment.get_water_vapour_mixing_ratio(), reshape=True
        )
        self.particulator.environment.get_predicted("thd").download(
            self.particulator.environment.get_thd(), reshape=True
        )
        self.solvers()
