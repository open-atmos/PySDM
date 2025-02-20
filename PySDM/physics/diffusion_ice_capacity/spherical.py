"""
capacity for approximation of ice crystals as spheres
"""


class Spherical:

    def __init__(self, _):
        pass

    @staticmethod
    def capacity(_, diameter, length=None):  # pylint: disable=unused-argument
        return diameter / 2.0
