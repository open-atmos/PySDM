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
        if choice not in ("Null", "Bolin1958", "MiyakeEtAl1968")
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
            isotope_diffusivity_ratios="HellmannAndHarvey2020",
        )
        const = formulae.constants
        temperature = 300 * si.K
        D = const.D0
        D_iso = (
            getattr(formulae.isotope_diffusivity_ratios, f"ratio_{iso}_heavy_to_light")(
                temperature
            )
            * D
        )
        vent_coeff = 1.01
        m_dm_dt = formulae.isotope_relaxation_timescale.isotope_m_dm_dt(
            rho_s=const.rho_w,
            radius=0.1 * si.mm,
            D_iso=vent_coeff * D_iso,
            D=D,
            S=1.01,
            R_liq=getattr(const, f"VSMOW_R_{iso}"),
            alpha=getattr(
                formulae.isotope_equilibrium_fractionation_factors, f"alpha_l_{iso}"
            )(temperature),
            R_vap=getattr(const, f"VSMOW_R_{iso}"),
            Fk=formulae.drop_growth.Fk(T=const.T_tri, K=const.K0, lv=const.l_tri),
        )
        sut = formulae.isotope_relaxation_timescale.tau

        # act
        result = sut(m_dm_dt)

        # assert
        assert result.check("[time]")
        assert 0 * si.s < result < 10 * si.s


def test_bolin_tritium_formula_unit():
    with DimensionalAnalysis():
        # arrange
        si = constants_defaults.si
        formulae = Formulae(
            isotope_relaxation_timescale="Bolin1958",
            constants={"BOLIN_ISOTOPE_TIMESCALE_COEFF_C1": 1 * si.dimensionless},
        )
        sut = formulae.isotope_relaxation_timescale.tau_of_rdrdt

        # act
        result = sut(radius=1 * si.um, r_dr_dt=1 * si.um**2 / si.s)

        # assert
        assert result.check("[time]")
