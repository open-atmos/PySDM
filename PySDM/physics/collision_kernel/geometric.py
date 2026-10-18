"""
basic geometric kernel
"""


class Geometric:  # pylint:disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def requested_attributes(_):
        return "radius", "relative fall velocity", ""

    @staticmethod
    def liquid_liquid(const, attr0_a, attr0_b, attr1_a, attr1_b, attr2_a, attr2_b):
        return (
            const.GEOMETRIC_collection_efficiency
            * const.PI
            * (attr0_a + attr0_b) ** 2
            * abs(attr1_a - attr1_b)
        )
