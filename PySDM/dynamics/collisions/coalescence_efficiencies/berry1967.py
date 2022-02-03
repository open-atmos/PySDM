"""
Created at 05.07.2020
"""

from ._parameterized import Parameterized


class Berry1967(Parameterized):
    """
    E.X. Berry 1967
    Cloud Droplet Growth by Collection
    """

    def __init__(self):
        super().__init__((1, 1, -27, 1.65, -58, 1.9, 15, 1.13, 16.7, 1, .004, 4, 8))
