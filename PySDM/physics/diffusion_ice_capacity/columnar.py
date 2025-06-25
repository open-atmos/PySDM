"""
capacity for columnar ice crystals approximated as prolate ellipsoids
eq. A11 & A12 in Spichtinger et al. 2023 (https://doi.org/10.5194/acp-23-2035-2023)
"""


class Columnar:  # pylint: disable=too-few-public-methods

    def __init__(self, _):
        pass

    @staticmethod
    def capacity(const, mass):
        return (
            const.capacity_columnar_ice_A1 * mass**const.capacity_columnar_ice_B1
            + const.capacity_columnar_ice_A2 * mass**const.capacity_columnar_ice_B2
        )
