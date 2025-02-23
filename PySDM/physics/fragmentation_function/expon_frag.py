"""
Formulae supporting `PySDM.dynamics.collisions.breakup_fragmentations.expon_frag`
"""

import warnings

from .exponential import Exponential


class ExponFrag(Exponential):  # pylint: disable=too-few-public-methods
    def __init_subclass__(cls):
        warnings.warn("Class has been renamed", DeprecationWarning)
