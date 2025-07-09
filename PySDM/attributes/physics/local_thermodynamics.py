from ..impl import register_attribute, CellAttribute, ExtensiveAttribute

NAME = "drop-local water vapour mixing ratio"


@register_attribute(
    name=NAME, variant=lambda dynamics, _: "DropLocalThermodynamics" in dynamics
)
class LocalVapourMixingRatioRelaxed(
    ExtensiveAttribute
):  # TODO: it should not be extensive - what happens with Qk and Tk when two droplets collide!
    def __init__(self, builder):
        super().__init__(builder, name=NAME)


@register_attribute(
    name=NAME,
    variant=lambda dynamics, _: "DropLocalThermodynamics" not in dynamics,
)
class LocalVapourMixingEqualToAmbientOne(CellAttribute):
    def __init__(self, builder):
        super().__init__(builder, name=NAME)

    # TODO: provide a way to initialise with cell values

    def recalculate(self):
        pass
        # TODO: just copy from the environment (but after nv forcing!)
        # TODO: maybe use this attr to track the tendency?


# TODO: also for temperature (or use the n_dimensional vector, or make a factory and loop)
# TODO: shouldn't it be defined as mass rather than mixing ratio? (like for multiplicity)
