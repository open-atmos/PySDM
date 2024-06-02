# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Builder
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.products import ParticleConcentration, TotalParticleConcentration

N_SD = 11
DV = 22 * si.m**3
MULTIPLICITY = 33
DROP_VOLUME = 44 * si.um**3
RHOD = 1.55 * si.kg / si.m**3
ENV = Box(dt=0, dv=DV)
ATTRIBUTES = {
    "multiplicity": np.asarray([MULTIPLICITY] * N_SD),
    "volume": np.asarray([DROP_VOLUME] * N_SD),
}
CONC = N_SD * MULTIPLICITY / DV


class TestParticleConcentration:
    @staticmethod
    @pytest.mark.parametrize("stp", (True, False))
    def test_stp(backend_instance, stp):
        # arrange
        builder = Builder(n_sd=N_SD, backend=backend_instance, environment=ENV)
        particulator = builder.build(
            attributes=ATTRIBUTES, products=(TotalParticleConcentration(stp=stp),)
        )
        if stp:
            ENV["rhod"] = RHOD

        # act
        prod = particulator.products["total particle concentration"].get()

        # assert
        np.testing.assert_approx_equal(
            actual=prod,
            desired=(
                CONC / RHOD * particulator.formulae.constants.rho_STP if stp else CONC
            ),
            significant=7,
        )

    @staticmethod
    @pytest.mark.parametrize("specific", (True, False))
    def test_specific(backend_instance, specific):
        # arrange
        builder = Builder(n_sd=N_SD, backend=backend_instance, environment=ENV)
        particulator = builder.build(
            attributes=ATTRIBUTES, products=(ParticleConcentration(specific=specific),)
        )
        if specific:
            ENV["rhod"] = RHOD

        # act
        prod = particulator.products["particle concentration"].get()

        # assert
        np.testing.assert_approx_equal(
            actual=prod, desired=CONC / RHOD if specific else CONC, significant=7
        )
