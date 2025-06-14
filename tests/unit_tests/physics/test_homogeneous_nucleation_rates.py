"""
test for homogeneous nucleation rate parameterisations
"""

from contextlib import nullcontext
import re
import pytest
from matplotlib import pyplot
import numpy as np
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
