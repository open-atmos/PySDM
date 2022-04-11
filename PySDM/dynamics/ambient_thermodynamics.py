"""
environment-sync triggering class
"""


class AmbientThermodynamics:
    def __init__(self):
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

    def __call__(self):
        self.particulator.environment.sync()
