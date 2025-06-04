"""
capacity for columnar ice crystals approximated as prolate ellipsoids
fit for Eq. 32 in Spichtinger & Gierens 2009 (https://doi.org/10.5194/acp-9-685-2009)
"""


class Columnar:  # pylint: disable=too-few-public-methods

    def __init__(self, _):
        pass

    @staticmethod
    def capacity(const, mass):
        return (const.capacity_columnar_ice_A1 * mass**const.capacity_columnar_ice_B1
                + const.capacity_columnar_ice_A2 * mass**const.capacity_columnar_ice_B2
                )

