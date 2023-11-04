"""
GPU-resident backend using NVRTC runtime compilation library for CUDA
"""

import os
import warnings

from PySDM.backends.impl_thrust_rtc.methods.collisions_methods import CollisionsMethods
from PySDM.backends.impl_thrust_rtc.methods.condensation_methods import (
    CondensationMethods,
)
from PySDM.backends.impl_thrust_rtc.methods.displacement_methods import (
    DisplacementMethods,
)
from PySDM.backends.impl_thrust_rtc.methods.freezing_methods import FreezingMethods
from PySDM.backends.impl_thrust_rtc.methods.moments_methods import MomentsMethods
from PySDM.backends.impl_thrust_rtc.methods.physics_methods import PhysicsMethods
from PySDM.backends.impl_thrust_rtc.methods.terminal_velocity_methods import (
    TerminalVelocityMethods,
)
from PySDM.formulae import Formulae
from PySDM.storages.holders.thrust_rtc import ThrustRTCStorageHolder
from PySDM.storages.thrust_rtc.backend.index import IndexBackend
from PySDM.storages.thrust_rtc.backend.pair import PairBackend
from PySDM.storages.thrust_rtc.conf import trtc


class ThrustRTC(  # pylint: disable=duplicate-code,too-many-ancestors
    ThrustRTCStorageHolder,
    CollisionsMethods,
    PairBackend,
    IndexBackend,
    PhysicsMethods,
    CondensationMethods,
    MomentsMethods,
    DisplacementMethods,
    TerminalVelocityMethods,
    FreezingMethods,
):
    ENABLE = True

    default_croupier = "global"

    def __init__(
        self, formulae=None, double_precision=False, debug=False, verbose=False
    ):
        super().__init__(double_precision)
        self.formulae = formulae or Formulae()

        CollisionsMethods.__init__(self)
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
