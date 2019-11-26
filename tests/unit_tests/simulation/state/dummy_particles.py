"""
Created at 19.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class DummyParticles:

    def __init__(self, backend, n_sd, dt):
        self.backend = backend
        self.n_sd = n_sd
        self.dt = dt
        self.environment = None

    def set_environment(self, environment_class, params):
        self.environment = environment_class(None, *params)
