""" tests for optical formulae (albedo, optical depth, ...) """

import pytest

from PySDM import Formulae
from PySDM.physics import optical_albedo, optical_depth, constants_defaults
from PySDM.formulae import _choices
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestOptical:
    @staticmethod
    @pytest.mark.parametrize(
        "paper", [p for p in _choices(optical_albedo) if p != "Null"]
    )
    def test_albedo_unit(paper):
        with DimensionalAnalysis():
            # arrange
            formulae = Formulae(optical_albedo=paper)
            si = constants_defaults.si
            tau = 1 * si.dimensionless

            # act
            albedo = formulae.optical_albedo.albedo(tau)

            # assert
            assert albedo.check("[]")

    @staticmethod
    @pytest.mark.parametrize(
        "paper", [p for p in _choices(optical_depth) if p != "Null"]
    )
    def test_optical_depth_unit(paper):
        with DimensionalAnalysis():
            # arrange
            formulae = Formulae(optical_depth=paper)
            si = constants_defaults.si
            liquid_water_path = 30 * si.g / si.m**2
            effective_radius = 10 * si.um

            # act
            tau = formulae.optical_depth.tau(
                LWP=liquid_water_path, reff=effective_radius
            )

            # assert
            assert tau.check("[]")
