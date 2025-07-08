from PySDM import register_attribute
from PySDM.attributes.impl import DerivedAttribute


@register_attribute(
    name="dry volume", variant=lambda dynamics, _: "DropLocalThermodynamics" in dynamics
)
class LocalVapourMixingRatioRelaxed(DerivedAttribute):
    pass


@register_attribute(
    name="dry volume",
    variant=lambda dynamics, _: "DropLocalThermodynamics" not in dynamics,
)
class LocalVapourMixingEqualToAmbientOne(DerivedAttribute):
    pass


# TODO: also for temperature (or use the n_dimensional vector, or make a factory and loop)
