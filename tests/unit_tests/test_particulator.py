# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

from .dummy_particulator import DummyParticulator


class TestParticulator:
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

    @staticmethod
    @pytest.mark.parametrize("isotopes", (("1H",), ("1H", "2H"), ("16O", "17I", "18O")))
    def test_isotopic_fractionation_marks_moles_as_updated(
        backend_class, isotopes: tuple
    ):
        # arrange
        class AttributesMock:
            def __init__(self):
                self.updated = []

            def __getitem__(self, item):
                return

            def mark_updated(self, attr):
                self.updated += [attr]

        class DP(DummyParticulator):
            pass

        particulator = DP(backend_class, 44)
        particulator.attributes = AttributesMock()

        # act
        particulator.isotopic_fractionation(heavy_isotopes=isotopes)

        # assert
        assert particulator.attributes.updated == [
            f"moles_{isotope}" for isotope in isotopes
        ]
