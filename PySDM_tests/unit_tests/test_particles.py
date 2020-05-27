"""
Created at 27.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM_tests.unit_tests.state.dummy_particles import DummyParticles
from PySDM.backends.default import Default


class TestParticles:

    def test_observer(self):
        class Observer:
            def __init__(self, particles):
                self.steps = 0
                self.particles = particles
                self.particles.observers.append(self)

            def notify(self):
                self.steps += 1
                assert self.steps == self.particles.n_steps

        steps = 33
        particles = DummyParticles(Default, 44)
        observer = Observer(particles)
        particles.run(steps)

        assert observer.steps == steps
