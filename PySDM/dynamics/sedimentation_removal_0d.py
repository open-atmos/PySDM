"""deposition removal logic for zero-dimensional environments"""

from typing import Optional
import numpy as np
from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class SedimentationRemoval0D:
    def __init__(self, *, stochastic_sedimentation_removal: Optional[bool] = True):
        """stochastic or deterministic removal"""
        self.stochastic_sedimentation_removal = stochastic_sedimentation_removal
        self.particulator = None

    def register(self, builder):
        builder.request_attribute("relative fall velocity")
        assert builder.particulator.environment.mesh.n_dims == 0
        self.particulator = builder.particulator

    def __call__(self):
        """for stochastic removal, see, e.g., the naive scheme in Algorithm 1 in
        [Curtis et al. 2016](https://doi.org/10.1016/j.jcp.2016.06.029)"""

        self.particulator.sedimentation_removal(
            stochastic_sedimentation_removal=self.stochastic_sedimentation_removal,
            length_scale=np.cbrt(self.particulator.environment.mesh.dv),
        )
