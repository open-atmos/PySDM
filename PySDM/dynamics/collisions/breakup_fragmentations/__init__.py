"""
TODO #744
"""
import warnings

from .always_n import AlwaysN
from .constant_mass import ConstantMass
from .exponential import Exponential
from .feingold1988 import Feingold1988
from .gaussian import Gaussian
from .lowlist82 import LowList1982Nf
from .slams import SLAMS
from .straub2010 import Straub2010Nf


class ExponFrag(Exponential):  # pylint: disable=too-few-public-methods
    def __init_subclass__(cls):
        warnings.warn("Class has been renamed", DeprecationWarning)
