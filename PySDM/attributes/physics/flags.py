from ..impl import register_attribute, FlagAttribute


@register_attribute()
class FlagCoalescence(FlagAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="coalescence flag")
