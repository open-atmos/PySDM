"""
Created at 01.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from ._storage_methods import StorageMethods
from ._maths_methods import MathsMethods
from ._algorithmic_methods import AlgorithmicMethods


class ThrustRTC(StorageMethods, MathsMethods, AlgorithmicMethods):
    pass
