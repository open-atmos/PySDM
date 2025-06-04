"""
capacity for columnar ice crystals approximated as prolate ellipsoids
fit for Eq. 32 in Spichtinger & Gierens 2009 (https://doi.org/10.5194/acp-9-685-2009)
"""


class Columnar:  # pylint: disable=too-few-public-methods

    def __init__(self, _):
        pass

    @staticmethod
    def capacity(const, mass):
        return (const.columnar_capacity_a1 * mass**const.columnar_capacity_b1
                + const.columnar_capacity_a2 * mass**const.columnar_capacity_b2
                )

