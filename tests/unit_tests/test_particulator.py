# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from .dummy_particulator import DummyParticulator


class TestParticulator:  # pylint: disable=too-few-public-methods
    @staticmethod
    def test_observer(backend_class):
        class Observer:  # pylint: disable=too-few-public-methods
            def __init__(self, particulator):
                self.steps = 0
                self.particulator = particulator
                self.particulator.observers.append(self)

            def notify(self):
                self.steps += 1
                assert self.steps == self.particulator.n_steps

        steps = 33
        particulator = DummyParticulator(backend_class, 44)
        observer = Observer(particulator)
        particulator.run(steps)

        assert observer.steps == steps
