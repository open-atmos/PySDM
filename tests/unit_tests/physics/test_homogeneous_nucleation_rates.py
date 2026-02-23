"""
test for homogeneous nucleation rate parameterisations
"""

from contextlib import nullcontext
import re
import pytest
from matplotlib import pyplot
import numpy as np
from scipy.optimize import elementwise
from PySDM.formulae import Formulae, _choices
from PySDM.physics import homogeneous_ice_nucleation_rate
from PySDM import physics
from PySDM.physics.dimensional_analysis import DimensionalAnalysis

SPICHTINGER_ET_AL_2023_FIG2_DATA = {
    "da_w_ice": [0.27, 0.29, 0.31, 0.33],
    "jhom_log10": [5, 11, 15, 20],
}


class TestHomogeneousIceNucleationRate:
    @staticmethod
    @pytest.mark.parametrize(
        "index", range(len(SPICHTINGER_ET_AL_2023_FIG2_DATA["da_w_ice"]))
    )
    @pytest.mark.parametrize(
        "parametrisation, context",
        (
            ("Koop_Correction", nullcontext()),
            (
                "Koop2000",
                pytest.raises(
                    AssertionError, match="Items are not equal to 2 significant digits"
                ),
            ),
            (
                "KoopMurray2016",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "x and y must have same first dimension, but have shapes (4,) and (1,)"
                    ),
                ),
            ),
        ),
    )
    def test_fig_2_in_spichtinger_et_al_2023(
        index, parametrisation, context, plot=False
    ):
        """Fig. 2 in [Spichtinger et al. 2023](https://doi.org/10.5194/acp-23-2035-2023)"""
        # arrange
        formulae = Formulae(
            homogeneous_ice_nucleation_rate=parametrisation,
            saturation_vapour_pressure="MurphyKoop2005",
        )

        # act
        with context:
            jhom_log10 = np.log10(
                formulae.homogeneous_ice_nucleation_rate.j_hom(
                    np.nan, np.asarray(SPICHTINGER_ET_AL_2023_FIG2_DATA["da_w_ice"])
                )
            )

            # plot
            pyplot.scatter(
                x=[SPICHTINGER_ET_AL_2023_FIG2_DATA["da_w_ice"][index]],
                y=[SPICHTINGER_ET_AL_2023_FIG2_DATA["jhom_log10"][index]],
                color="red",
                marker="x",
            )
            pyplot.plot(
                SPICHTINGER_ET_AL_2023_FIG2_DATA["da_w_ice"],
                jhom_log10,
                marker=".",
            )
            pyplot.gca().set(
                xlabel=r"water activity difference $\Delta a_w$",
                ylabel="log$_{10}(J)$",
                title=parametrisation,
                xlim=(0.26, 0.34),
                ylim=(0, 25),
            )
            pyplot.grid()
            if plot:
                pyplot.show()
            else:
                pyplot.clf()

            # assert
            np.testing.assert_approx_equal(
                actual=jhom_log10[index],
                desired=SPICHTINGER_ET_AL_2023_FIG2_DATA["jhom_log10"][index],
                significant=2,
            )

    @staticmethod
    def test_fig_11_in_spichtinger_et_al_2023(plot=False):
        """Fig. 11 in [Spichtinger et al. 2023](https://doi.org/10.5194/acp-23-2035-2023)"""
        # arrange
        formulae = Formulae(
            homogeneous_ice_nucleation_rate="KoopMurray2016",
            saturation_vapour_pressure="MurphyKoop2005",
        )
        formulae_KoopMurray2016_DAW = Formulae(
            homogeneous_ice_nucleation_rate="KoopMurray2016_DWA",
        )
        formulae_Koop_Correction = Formulae(
            homogeneous_ice_nucleation_rate="Koop_Correction",
        )

        d_aw_ice = np.arange(0.24, 0.37, 0.01)
        S_i = -1.0 / (d_aw_ice - 1.0)

        def sat_T(temperature, S_i=1.5):
            return (
                formulae.saturation_vapour_pressure.pvs_water(temperature)
                / formulae.saturation_vapour_pressure.pvs_ice(temperature)
                - S_i
            )

        temperature = elementwise.find_root(sat_T, (210.0, 260.0), args=(S_i,)).x

        # act
        J_hom_parametrisations = {
            "Koop & Murray (2016)": np.log10(
                formulae.homogeneous_ice_nucleation_rate.j_hom(temperature, d_aw_ice)
            ),
            "quadratic fit": np.log10(
                formulae_KoopMurray2016_DAW.homogeneous_ice_nucleation_rate.j_hom(
                    temperature, d_aw_ice
                )
            ),
            "Koop et al. (2000), corr.": np.log10(
                formulae_Koop_Correction.homogeneous_ice_nucleation_rate.j_hom(
                    temperature, d_aw_ice
                )
            ),
        }

        # plot
        for parametrisation, data in J_hom_parametrisations.items():
            pyplot.plot(
                d_aw_ice,
                data,
                label=parametrisation,
            )
        pyplot.gca().set(
            xlabel=r"water activity difference $\Delta a_w$",
            ylabel="log$_{10}(J)$",
            xlim=(0.24, 0.36),
            ylim=(-10, 30),
        )
        pyplot.grid()
        pyplot.legend()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        index = range(
            np.where(np.isclose(d_aw_ice, 0.27))[0][0],
            np.where(np.isclose(d_aw_ice, 0.34))[0][0] + 1,
        )
        np.testing.assert_allclose(
            np.log10(J_hom_parametrisations["Koop & Murray (2016)"][index]),
            np.log10(J_hom_parametrisations["quadratic fit"][index]),
            rtol=0.01,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "pvs_parametrisation, context",
        (
            (
                "FlatauWalkoCotton",
                pytest.raises(
                    AssertionError, match="Not equal to tolerance rtol=0.05, atol=0"
                ),
            ),
            ("MurphyKoop2005", nullcontext()),
        ),
    )
    def test_fig_1_in_spichtinger_et_al_2023(
        pvs_parametrisation, context, plot=False
    ):  # pylint: disable=too-many-locals
        # arrange
        si = physics.si
        formulae = Formulae(
            homogeneous_ice_nucleation_rate="KoopMurray2016",
            saturation_vapour_pressure=pvs_parametrisation,
        )
        formulae_koop2000 = Formulae(
            homogeneous_ice_nucleation_rate="Koop2000",
        )
        formulae_Koop_Correction = Formulae(
            homogeneous_ice_nucleation_rate="Koop_Correction",
        )
        temperature = np.linspace(230, 245, num=16) * si.K

        with context:
            # act
            pv_sat_water = (
                formulae.saturation_vapour_pressure.pvs_water(temperature) * si.Pa
            )
            pv_sat_ice = (
                formulae.saturation_vapour_pressure.pvs_ice(temperature) * si.Pa
            )

            d_aw_ice = (pv_sat_water * 1.0 / pv_sat_ice - 1) * pv_sat_ice / pv_sat_water

            J_hom_parametrisations = {
                "KoopMurray2016": np.log10(
                    formulae.homogeneous_ice_nucleation_rate.j_hom(
                        temperature, d_aw_ice
                    )
                ),
                "Koop2000": np.log10(
                    formulae_koop2000.homogeneous_ice_nucleation_rate.j_hom(
                        temperature, d_aw_ice
                    )
                ),
                "Koop_Correction": np.log10(
                    formulae_Koop_Correction.homogeneous_ice_nucleation_rate.j_hom(
                        temperature, d_aw_ice
                    )
                ),
            }

            # plot
            J_hom_range = (-10, 25)
            for parametrisation, data in J_hom_parametrisations.items():
                pyplot.plot(
                    temperature,
                    data,
                    label=parametrisation,
                )
            pyplot.grid()
            pyplot.gca().set(
                xlabel=r"temperature (K)",
                ylabel="log$_{10}(J)$",
                title="saturation_vapour_pressure: " + pvs_parametrisation,
                xlim=(np.amin(temperature), np.amax(temperature)),
                ylim=J_hom_range,
            )
            pyplot.xticks(ticks=temperature, minor=True)
            pyplot.xticks(ticks=np.arange(230, 250, 5), minor=False)
            pyplot.yticks(
                ticks=np.arange(J_hom_range[0], J_hom_range[1] + 1, 1), minor=True
            )
            pyplot.legend()

            if plot:
                pyplot.show()
            else:
                pyplot.clf()

            # Assert
            assert all(
                J_hom_parametrisations["Koop2000"]
                >= J_hom_parametrisations["KoopMurray2016"]
            )
            assert all(
                J_hom_parametrisations["Koop2000"]
                >= J_hom_parametrisations["Koop_Correction"]
            )
            index = range(
                np.where(temperature == 235)[0][0],
                np.where(temperature == 240)[0][0] + 1,
            )
            np.testing.assert_allclose(
                J_hom_parametrisations["Koop_Correction"][index],
                J_hom_parametrisations["KoopMurray2016"][index],
                rtol=0.05,
            )

    @staticmethod
    @pytest.mark.parametrize("variant", _choices(homogeneous_ice_nucleation_rate))
    def test_units(variant):
        if variant == "Null":
            pytest.skip()

        with DimensionalAnalysis():
            # arrange
            si = physics.si
            formulae = Formulae(
                homogeneous_ice_nucleation_rate=variant,
                constants=(
                    {} if variant != "Constant" else {"J_HOM": 1 / si.m**3 / si.s}
                ),
            )
            sut = formulae.homogeneous_ice_nucleation_rate
            temperature = 250 * si.K
            da_w_ice = 0.3 * si.dimensionless

            # act
            value = sut.j_hom(temperature, da_w_ice)

            # assert
            assert value.check("1/[volume]/[time]")
