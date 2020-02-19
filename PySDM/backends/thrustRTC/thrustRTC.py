"""
Created at 01.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from ._storage_methods import StorageMethods
from ._maths_methods import MathsMethods
from ._special_methods import SpecialMethods


class ThrustRTC(StorageMethods, MathsMethods, SpecialMethods):
    pass
