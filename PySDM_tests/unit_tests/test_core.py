from PySDM_tests.unit_tests.dummy_core import DummyCore
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


class TestCore:

    @staticmethod
    def test_observer(backend):
        class Observer:
            def __init__(self, core):
                self.steps = 0
                self.core = core
                self.core.observers.append(self)

            def notify(self):
                self.steps += 1
                assert self.steps == self.core.n_steps

        steps = 33
        core = DummyCore(backend, 44)
        observer = Observer(core)
        core.run(steps)

        assert observer.steps == steps
