"""
Created at 27.05.2020
"""

from PySDM_tests.unit_tests.dummy_core import DummyCore
from PySDM.backends.default import Default


class TestCore:

    def test_observer(self):
        class Observer:
            def __init__(self, core):
                self.steps = 0
                self.core = core
                self.core.observers.append(self)

            def notify(self):
                self.steps += 1
                assert self.steps == self.core.n_steps

        steps = 33
        core = DummyCore(Default, 44)
        observer = Observer(core)
        core.run(steps)

        assert observer.steps == steps
