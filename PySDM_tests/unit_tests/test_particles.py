"""
Created at 27.05.2020
"""

from PySDM_tests.unit_tests.state.dummy_core import DummyCore
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
        particles = DummyCore(Default, 44)
        observer = Observer(particles)
        particles.run(steps)

        assert observer.steps == steps
