# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from collections import namedtuple

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

    @staticmethod
    def test_seeding_marks_modified_attributes_as_updated(backend_class):
        # arrange
        storage = backend_class().Storage.empty(1, dtype=int)
        abc = ["a", "b", "c"]

        class ParticleAttributes:
            def __init__(self):
                self.updated = []
                self.super_droplet_count = -1
                self.__idx = storage  # pylint: disable=unused-private-member
                self.idx_reset = False
                self.sane = False

            def get_extensive_attribute_storage(self):
                return storage

            def get_extensive_attribute_keys(self):
                return abc

            def __getitem__(self, item):
                return storage

            def mark_updated(self, attr):
                self.updated += [attr]

            def reset_idx(self):
                self.idx_reset = True

            def sanitize(self):
                self.sane = True

        class DP(DummyParticulator):
            pass

        particulator = DP(backend_class, 44)
        particulator.attributes = ParticleAttributes()
        # fmt: off
        particulator.backend.seeding = (
            lambda
                idx,
                multiplicity,
                extensive_attributes,
                seeded_particle_index,
                seeded_particle_multiplicity,
                seeded_particle_extensive_attributes,
                number_of_super_particles_to_inject:
            None
        )
        # fmt: on

        # act
        particulator.seeding(
            seeded_particle_index=storage,
            seeded_particle_multiplicity=storage,
            seeded_particle_extensive_attributes=storage,
            number_of_super_particles_to_inject=0,
        )

        # assert
        assert particulator.attributes.updated == ["multiplicity"] + abc
        assert particulator.attributes.idx_reset
        assert particulator.attributes.sane

    @staticmethod
    def test_seeding_fails_if_no_null_super_droplets_availale(backend_class):
        # arrange
        a_number = 44

        particulator = DummyParticulator(backend_class, n_sd=a_number)
        storage = backend_class().Storage.empty(1, dtype=int)

        particulator.attributes = namedtuple(
            typename="_", field_names=("super_droplet_count",)
        )(super_droplet_count=a_number)

        # act
        with pytest.raises(ValueError, match="No available seeds to inject"):
            particulator.seeding(
                seeded_particle_index=storage,
                seeded_particle_multiplicity=storage,
                seeded_particle_extensive_attributes=storage,
                number_of_super_particles_to_inject=0,
            )
