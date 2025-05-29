"""
GPU-resident backend using NVRTC runtime compilation library for CUDA
"""

import os
import warnings

import numpy as np

from PySDM.backends.impl_thrust_rtc.conf import trtc
from PySDM.backends.impl_thrust_rtc.methods.collisions_methods import CollisionsMethods
from PySDM.backends.impl_thrust_rtc.methods.condensation_methods import (
    CondensationMethods,
)
from PySDM.backends.impl_thrust_rtc.methods.displacement_methods import (
    DisplacementMethods,
)
from PySDM.backends.impl_thrust_rtc.methods.freezing_methods import FreezingMethods
from PySDM.backends.impl_thrust_rtc.methods.index_methods import IndexMethods
from PySDM.backends.impl_thrust_rtc.methods.isotope_methods import IsotopeMethods
from PySDM.backends.impl_thrust_rtc.methods.moments_methods import MomentsMethods
from PySDM.backends.impl_thrust_rtc.methods.pair_methods import PairMethods
from PySDM.backends.impl_thrust_rtc.methods.physics_methods import PhysicsMethods
from PySDM.backends.impl_thrust_rtc.methods.terminal_velocity_methods import (
    TerminalVelocityMethods,
)
from PySDM.backends.impl_thrust_rtc.random import Random as ImportedRandom
from PySDM.backends.impl_thrust_rtc.storage import make_storage_class
from PySDM.formulae import Formulae


class ThrustRTC(  # pylint: disable=duplicate-code,too-many-ancestors
    CollisionsMethods,
    PairMethods,
    IndexMethods,
    PhysicsMethods,
    CondensationMethods,
    MomentsMethods,
    DisplacementMethods,
    TerminalVelocityMethods,
    FreezingMethods,
    IsotopeMethods,
):
    ENABLE = True
    Random = ImportedRandom

    default_croupier = "global"

    def __init__(
        self, formulae=None, double_precision=False, debug=False, verbose=False
    ):
        self.formulae = formulae or Formulae()

        self._conv_function = trtc.DVDouble if double_precision else trtc.DVFloat
        self._real_type = "double" if double_precision else "float"
        self._np_dtype = np.float64 if double_precision else np.float32

        self.Storage = make_storage_class(self)

        CollisionsMethods.__init__(self)
        PairMethods.__init__(self)
        IndexMethods.__init__(self)
        PhysicsMethods.__init__(self)
        CondensationMethods.__init__(self)
        MomentsMethods.__init__(self, double_precision=double_precision)
        DisplacementMethods.__init__(self)
        TerminalVelocityMethods.__init__(self)
        FreezingMethods.__init__(self)

        trtc.Set_Kernel_Debug(debug)
        trtc.Set_Verbose(verbose)

        if not ThrustRTC.ENABLE and "CI" not in os.environ:
            warnings.warn("CUDA is not available, using FakeThrustRTC!")
