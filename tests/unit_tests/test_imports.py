""" test ensuring that all needed __init__.py entries are in place """

import pytest

import PySDM

CLASSES = (
    "Builder",
    "Formulae",
    "Particulator",
    "attributes.chemistry.Acidity",
    "attributes.physics.DryVolume",
    "backends.CPU",
    "backends.GPU",
    "dynamics.Condensation",
    "dynamics.collisions.breakup_fragmentations.AlwaysN",
    "dynamics.collisions.coalescence_efficiencies.LowList1982Ec",
    "dynamics.collisions.collision_kernels.Golovin",
    "dynamics.terminal_velocity.RogersYau",
    "environments.Box",
    "environments.Kinematic1D",
    "environments.Kinematic2D",
    "environments.Parcel",
    "exporters.VTKExporter",
    "initialisation.aerosol_composition.DryAerosolMixture",
    "initialisation.init_fall_momenta",
    "initialisation.sampling.spectral_sampling.DeterministicSpectralSampling",
    "initialisation.spectra.Lognormal",
    "physics.constants_defaults",
    "physics.diffusion_thermics.LoweEtAl2019",
    "physics.si",
    "products.size_spectral.EffectiveRadius",
)


class TestImports:
    @staticmethod
    @pytest.mark.parametrize("obj_path", CLASSES)
    def test_imports(obj_path):
        """one can import PySDM and then access classes from submodules,
        like PySDM.environments.Box, etc"""
        obj = PySDM
        for attr in obj_path.split("."):
            obj = getattr(obj, attr)

    @staticmethod
    def test_classes_sorted():
        assert tuple(sorted(CLASSES)) == CLASSES
