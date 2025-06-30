from PySDM.attributes.impl import register_attribute, DerivedAttribute


@register_attribute()
class DiffusionalGrowthMassChange(DerivedAttribute):
    def __init__(self, builder):
        attr = builder.get_attribute("water mass")
        super().__init__(
            builder, name="diffusional growth mass change", dependencies=(attr,)
        )

    def recalculate(self):
        pass  # TODO
