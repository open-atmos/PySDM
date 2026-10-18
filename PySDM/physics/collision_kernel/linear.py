"""
TODO #744
"""


class Linear:  # pylint:disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def requested_attributes(_):
        return "volume", "", ""

    # pylint:disable=unused-argument
    @staticmethod
    def liquid_liquid(const, attr0_a, attr0_b, attr1_a, attr1_b, attr2_a, attr2_b):
        return (attr0_a + attr0_b) * const.KERNEL_LINEAR_b + const.KERNEL_LINEAR_a
