"""
capacity for approximation of ice crystals as spheres
"""


class Spherical:

    def __init__(self, _):
        pass

    @staticmethod
    def capacity(const, diameter):
        return diameter / const.TWO
