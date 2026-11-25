"""
wrapper class for triggering integration in the Eulerian advection solver
"""

from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class EulerianAdvection:
    def __init__(self):
        self.particulator = None

    def register(self, particulator):
        self.particulator = particulator

    def __call__(self):
        for field in ("water_vapour_mixing_ratio", "thd"):
            self.particulator.environment.get_predicted(field).download(
                getattr(self.particulator.environment, f"get_{field}")(), reshape=True
            )
        self.particulator.environment.solvers(
            self.particulator.dynamics["Displacement"]
        )
