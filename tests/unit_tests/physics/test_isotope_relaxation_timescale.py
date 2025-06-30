"""
tests for isotope relaxation timescale formulae
"""

import pytest

from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.formulae import Formulae, _choices
from PySDM.physics import constants_defaults, isotope_relaxation_timescale


class TestIsotopeRelaxationTimescale:
    @staticmethod
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
                isotope_diffusivity_ratios="HellmannAndHarvey2020",
            )
            const = formulae.constants
            temperature = 300 * si.K
            D = const.D0
            D_iso = (
                getattr(
                    formulae.isotope_diffusivity_ratios, f"ratio_{iso}_heavy_to_light"
                )(temperature)
                * D
            )
            vent_coeff = 1
            dm_dt_over_m = formulae.isotope_relaxation_timescale.isotope_dm_dt_over_m(
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
            result = sut(dm_dt_over_m)

            # assert
            assert result.check("[time]")
            assert 0 * si.s < result.to_base_units() < 10 * si.s

    @staticmethod
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
            re = sut(
                formulae.isotope_relaxation_timescale.isotope_dm_dt_over_m(
                    dm_dt_over_m=1 / si.s
                )
            )

            # assert
            assert re.check("[time]")

    @staticmethod
    def test_bolin_number_unit():
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            formulae = Formulae(
                isotope_relaxation_timescale="Bolin1958",
                constants={"BOLIN_ISOTOPE_TIMESCALE_COEFF_C1": 1 * si.dimensionless},
            )
            sut = formulae.isotope_relaxation_timescale.bolin_number

            # act
            re = sut(
                diffusivity_ratio_heavy_to_light=1 * si.dimensionless,
                alpha=0.9 * si.dimensionless,
                rho_s=1 * si.kilograms / si.metres**3,
                Fd=1 * si.s / si.metres**2,
                Fk=1 * si.s / si.metres**2,
                saturation=0.5 * si.dimensionless,
                R_vap=1 * si.dimensionless,
                R_liq=1 * si.dimensionless,
            )

        assert re.check(si.dimensionless)
