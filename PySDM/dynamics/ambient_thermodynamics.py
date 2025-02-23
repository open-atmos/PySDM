"""
environment-sync triggering class
"""

from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class AmbientThermodynamics:
    def __init__(self):
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

    def __call__(self):
        self.particulator.environment.sync()
