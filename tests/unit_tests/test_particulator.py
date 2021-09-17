from .dummy_particulator import DummyParticulator
# noinspection PyUnresolvedReferences
from ..backends_fixture import backend


class TestParticulator:

    @staticmethod
    def test_observer(backend):
        class Observer:
            def __init__(self, particulator):
                self.steps = 0
                self.particulator = particulator
                self.particulator.observers.append(self)

            def notify(self):
                self.steps += 1
                assert self.steps == self.particulator.n_steps

        steps = 33
        particulator = DummyParticulator(backend, 44)
        observer = Observer(particulator)
        particulator.run(steps)

        assert observer.steps == steps
