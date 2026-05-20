import numpy as np
import pytest

from PySDM.physics import turbulent_relaxation_timescale, constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM import Formulae


class TestTurbulentRelaxationTimescale:
    @staticmethod
    @pytest.mark.parametrize(
        "constants", ({}, {"TURBULENT_RELAXATION_TIMESCALE_MULTIPLIER": 1})
    )
    def test_kolmogorov_inertial_range_scaling_and_unit(constants):
        with DimensionalAnalysis():
            # arrange
            formulae = Formulae(constants=constants)
            si = constants_defaults.si
            eddy_length_scale = 666 * si.m
            tke_dissipation_rate = 0.44e-3 * si.m**2 / si.s**3
            sut = turbulent_relaxation_timescale.KolmogorovInertialRangeScaled.tau

            # act
            tau = sut(formulae.constants, eddy_length_scale, tke_dissipation_rate)

            # assert
            if "TURBULENT_RELAXATION_TIMESCALE_MULTIPLIER" not in constants:
                assert np.isnan(tau)
            else:
                assert np.isfinite(tau)
            assert tau.check("[time]")
