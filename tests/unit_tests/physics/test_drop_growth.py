"""tests for drop growth formulae"""

import pytest
from matplotlib import pyplot
import numpy as np

from PySDM.formulae import _choices, Formulae
from PySDM.physics import drop_growth
from PySDM.physics.constants import PER_CENT, si, in_unit
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestDropGrowth:
    @staticmethod
    @pytest.mark.parametrize("paper", _choices(drop_growth))
    def test_unit(paper):
        """checks dimensionality of the returned value"""
        with DimensionalAnalysis():
            # arrange
            formulae = Formulae(drop_growth=paper)
            const = formulae.constants

            # act
            r_dr_dt = formulae.drop_growth.r_dr_dt(
                RH_eq=1,
                RH=1.05,
                Fk=formulae.drop_growth.Fk(T=const.T_tri, K=const.K0, lv=const.l_tri),
                Fd=formulae.drop_growth.Fd(T=const.T_tri, D=const.D0, pvs=const.p_tri),
            )

            # assert
            assert r_dr_dt.check("[area]/[time]")

    @staticmethod
    @pytest.mark.parametrize(
        ("paper_name", "error_range"),
        (
            ("Howell1949", (0.02, 0.03)),
            ("Mason1971", (-0.01, 0.01)),
            ("Fick", (0.5, 0.9)),
        ),
    )
    def test_fick_mason_1971_vs_1971_difference_vs_temperature(
        paper_name, error_range, plot=False
    ):
        """checks the relative difference between Mason's 1951 and 1971 formulae
        for a range of temperatures"""
        # arrange
        temperatures = Formulae().trivia.C2K(np.linspace(-10, 40) * si.K)
        papers = ("Howell1949", "Mason1971", "Fick")
        relative_error = {}
        # act
        formulae = {paper: Formulae(drop_growth=paper) for paper in papers}
        r_dr_dt = {
            paper: formulae[paper].drop_growth.r_dr_dt(
                RH_eq=1,
                RH=1.05,
                Fk=formulae[paper].drop_growth.Fk(
                    T=temperatures,
                    K=formulae[paper].constants.K0,
                    lv=formulae[paper].constants.l_tri,
                ),
                Fd=formulae[paper].drop_growth.Fd(
                    T=temperatures,
                    D=formulae[paper].constants.D0,
                    pvs=formulae[paper].constants.p_tri,
                ),
            )
            for paper in papers
        }

        for paper in papers:
            relative_error[paper] = r_dr_dt[paper] / r_dr_dt["Mason1971"] - 1

        # plot
        for paper in papers:
            pyplot.plot(
                temperatures, in_unit(relative_error[paper], PER_CENT), label=paper
            )
        pyplot.title("")
        pyplot.xlabel("temperature [K]")
        pyplot.ylabel("r dr/dt relative difference (vs. 1971) [%]")
        pyplot.grid()
        pyplot.legend()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        assert (abs(relative_error[paper_name]) > error_range[0]).all()
        assert (abs(relative_error[paper_name]) < error_range[1]).all()
