"""
Created at 01.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from ._methods import Methods
from ._algorithmic_methods import AlgorithmicMethods
from ._algorithmic_step_methods import AlgorithmicStepMethods
from ._storage_methods import StorageMethods
from ._maths_methods import MathsMethods
from ._physics_methods import PhysicsMethods
from .storage import Storage


class ThrustRTC(
    Methods,
    AlgorithmicMethods,
    AlgorithmicStepMethods,
    StorageMethods,
    MathsMethods,
    PhysicsMethods,
):
    storage = Storage
