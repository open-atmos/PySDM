"""
particle Reynolds number
"""

from ..impl import DerivedAttribute, register_attribute


@register_attribute(
    name="Reynolds number",
    variant=lambda _, formulae: formulae.ventilation.__name__ != "Neglect",
    dummy_default=True,
)
class ReynoldsNumber(DerivedAttribute):
    def __init__(self, particulator):
        self.radius = particulator.get_attribute("radius")
        self.velocity_wrt_air = particulator.get_attribute("relative fall velocity")
        self.cell_id = particulator.get_attribute("cell id")
        super().__init__(
            particulator,
            name="Reynolds number",
            dependencies=(self.radius, self.velocity_wrt_air, self.cell_id),
        )

    def recalculate(self):
        self.particulator.backend.reynolds_number(
            output=self.data,
            cell_id=self.cell_id.get(),
            dynamic_viscosity=self.particulator.environment["air dynamic viscosity"],
            density=self.particulator.environment["air density"],
            radius=self.radius.data,
            velocity_wrt_air=self.velocity_wrt_air.data,
        )
