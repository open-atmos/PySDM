"""
wrapper class for triggering integration in the Eulerian advection solver
"""

from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class EulerianAdvection:
    def __init__(self, solvers):
        self.solvers = solvers
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

    def __call__(self):
        for field in ("water_vapour_mixing_ratio", "thd"):
            self.particulator.environment.get_predicted(field).download(
                getattr(self.particulator.environment, f"get_{field}")(), reshape=True
            )
        self.solvers(self.particulator.dynamics["Displacement"])
