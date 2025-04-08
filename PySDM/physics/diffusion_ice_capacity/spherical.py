"""
capacity for approximation of ice crystals as spheres
"""


class Spherical:  # pylint: disable=too-few-public-methods

    def __init__(self, _):
        pass

    @staticmethod
    def capacity(const, diameter):
        return diameter / const.TWO
