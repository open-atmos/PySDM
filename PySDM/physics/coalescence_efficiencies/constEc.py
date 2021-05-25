"""
Created at 05.21.2021
"""

from ._parameterized import Parameterized


class ConstEc(Parameterized):

    def __init__(self, Ec=1.0):
        super().__init__((Ec, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))