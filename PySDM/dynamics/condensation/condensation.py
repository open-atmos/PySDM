"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from ...particles_builder import ParticlesBuilder
import numpy as np


default_rtol_x = 1e-8
default_rtol_thd = 1e-8


class Condensation:
    def __init__(self, particles_builder: ParticlesBuilder, kappa,
                 rtol_x=default_rtol_x,
                 rtol_thd=default_rtol_thd,
                 coord='volume logarithm', adaptive=True,
                 do_advection: bool = True,
                 do_condensation: bool = True,
                 ):
        self.particles = particles_builder.particles
        particles_builder._set_condensation_parameters(coord, adaptive)
        self.environment = self.particles.environment
        self.kappa = kappa
        self.rtol_x = rtol_x
        self.rtol_thd = rtol_thd

        self.do_advection = do_advection
        self.do_condensation = do_condensation

        self.substeps = self.particles.backend.array(self.particles.mesh.n_cell, dtype=int)
        self.substeps[:] = np.maximum(1, int(self.particles.dt))
        # TODO: reset substeps
        self.ripening_flags = self.particles.backend.array(self.particles.mesh.n_cell, dtype=int)
        self.particles.backend.fill(self.ripening_flags, 0)

    def __call__(self):
        if self.do_advection:
            self.environment.sync()
        if self.do_condensation:
            self.particles.condensation(
                kappa=self.kappa,
                rtol_x=self.rtol_x,
                rtol_thd=self.rtol_thd,
                substeps=self.substeps,
                ripening_flags=self.ripening_flags
            )

