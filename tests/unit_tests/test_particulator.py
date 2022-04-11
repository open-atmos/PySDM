# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from ..backends_fixture import backend_class
from .dummy_particulator import DummyParticulator

assert hasattr(backend_class, "_pytestfixturefunction")


class TestParticulator:
    @staticmethod
    # pylint: disable=redefined-outer-name
    def test_observer(backend_class):
        class Observer:
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
