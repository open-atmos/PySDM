"""deposition removal logic for zero-dimensional environments"""

from PySDM.dynamics.impl import register_dynamic
import numpy as np


@register_dynamic()
class DepositionRemoval:
    def __init__(self, *, all_or_nothing: bool):
        """stochastic ("all or nothing") or deterministic (multiplicity altering) removal"""
        self.all_or_nothing = all_or_nothing
        self.length_scale = None

    def register(self, builder):
        builder.request_attribute("relative fall velocity")
        assert builder.particulator.environment.mesh.n_dims == 0
        self.particulator = builder.particulator
        self.length_scale = np.cbrt(self.particulator.environment.mesh.dv)

    def __call__(self):
        """see, e.g., the naive scheme in Algorithm 1 in
        [Curtis et al. 2016](https://doi.org/10.1016/j.jcp.2016.06.029)"""
        self.particulator.deposition_removal(
            all_or_nothing=self.all_or_nothing,
            length_scale=self.length_scale,
        )
