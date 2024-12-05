"""
tests for isotope relaxation timescale formulae
"""

import pytest

from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.formulae import Formulae, _choices
from PySDM.physics import constants_defaults, isotope_relaxation_timescale


@pytest.mark.parametrize(
    "paper",
    [
        choice
        for choice in _choices(isotope_relaxation_timescale)
        if choice not in ("Null", "Bolin1958")
    ],
)
@pytest.mark.parametrize("iso", ("2H", "18O"))
def test_unit_and_magnitude(paper, iso):
    with DimensionalAnalysis():
        # arrange
        si = constants_defaults.si
        formulae = Formulae(
            isotope_relaxation_timescale=paper,
            isotope_equilibrium_fractionation_factors="HoritaAndWesolowski1994",
        )
        const = formulae.constants
        temperature = 300 * si.K
        M_iso = (
            getattr(const, f"M_{iso}")
            + const.M_1H
            + (const.M_1H if iso[-1] == "O" else const.M_16O)
        )
        sut = formulae.isotope_relaxation_timescale.tau
        alpha_iso = getattr(
            formulae.isotope_equilibrium_fractionation_factors, f"alpha_l_{iso}"
        )(temperature)
        e_s = formulae.saturation_vapour_pressure.pvs_water(temperature)
        radius = 0.1 * si.mm
        vent_coeff = 1.01

        # act
        result = sut(e_s, const.D0, M_iso, vent_coeff, radius, alpha_iso, temperature)

        # assert
        assert result.check("[time]")
        assert 1 * si.s < result < 10 * si.s


def test_bolin_tritium_formula_unit():
    with DimensionalAnalysis():
        # arrange
        si = constants_defaults.si
        formulae = Formulae(
            isotope_relaxation_timescale="Bolin1958",
            constants={"BOLIN_ISOTOPE_TIMESCALE_COEFF_C1": 1 * si.dimensionless},
        )
        sut = formulae.isotope_relaxation_timescale.tau

        # act
        result = sut(radius=si.um, r_dr_dt=si.um**2 / si.s)

        # assert
        assert result.check("[time]")
