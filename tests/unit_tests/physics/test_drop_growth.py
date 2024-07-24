""" tests for drop growth formulae """

import pytest
from matplotlib import pyplot
import numpy as np

from PySDM.formulae import _choices, Formulae
from PySDM.physics import drop_growth, constants_defaults, si, in_unit
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestDropGrowth:
    @staticmethod
    @pytest.mark.parametrize("paper", _choices(drop_growth))
    def test_unit(paper):
        """checks dimensionality of the returned value"""
        with DimensionalAnalysis():
            # arrange
            formulae = Formulae(drop_growth=paper)

            # act
            r_dr_dt = formulae.drop_growth.r_dr_dt(
                RH_eq=1,
                T=constants_defaults.T_tri,
                RH=1.05,
                lv=constants_defaults.l_tri,
                pvs=constants_defaults.p_tri,
                D=constants_defaults.D0,
                K=constants_defaults.K0,
                ventilation_factor=1,
            )

            # assert
            assert r_dr_dt.check("[area]/[time]")

    @staticmethod
    def test_mason_1971_vs_1951_difference_vs_temperature(plot=False):
        """checks the relative difference between Mason's 1951 and 1971 formulae
        for a range of temperatures"""
        # arrange
        temperatures = constants_defaults.T0 + np.linspace(-10, 40) * si.K
        papers = ("Mason1951", "Mason1971")

        # act
        formulae = {paper: Formulae(drop_growth=paper) for paper in papers}
        r_dr_dt = {
            paper: formulae[paper].drop_growth.r_dr_dt(
                RH_eq=1,
                T=temperatures,
                RH=1.05,
                lv=constants_defaults.l_tri,
                pvs=constants_defaults.p_tri,
                D=constants_defaults.D0,
                K=constants_defaults.K0,
                ventilation_factor=1,
            )
            for paper in papers
        }
        relative_error = r_dr_dt["Mason1971"] / r_dr_dt["Mason1951"] - 1

        # plot
        pyplot.plot(temperatures, in_unit(relative_error, constants_defaults.PER_CENT))
        pyplot.title("")
        pyplot.xlabel("temperature [K]")
        pyplot.ylabel("r dr/dt relative difference (1971 vs. 1951) [%]")
        pyplot.grid()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        (relative_error < 0.03).all()
        (relative_error > 0.02).all()
        (np.diff(relative_error) < 0).all()
