# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import inspect

import numpy as np
from matplotlib import pyplot

from PySDM import Formulae
from PySDM.physics import si


class TestLatentHeat:
    @staticmethod
    def test_latent_vapourisation(plot=False):
        # Arrange
        formulae = {
            k: Formulae(latent_heat_vapourisation=k) for k in ("Kirchhoff", "Lowe2019")
        }
        const = Formulae().constants
        temperature = np.linspace(-20, 20) + const.T_tri

        # Plot
        pyplot.axhline(const.l_tri, label="triple point", color="red")
        pyplot.axvline(const.T_tri, color="red")
        for key, val in formulae.items():
            for name, func in inspect.getmembers(val.latent_heat_vapourisation):
                if name[:2] not in ("__", "a_"):
                    pyplot.plot(temperature, func(temperature), label=f"{key}::{name}")
        pyplot.grid()
        pyplot.legend()
        pyplot.xlabel("T [K]")
        pyplot.ylabel("Lv [J/kg]")
        if plot:
            pyplot.show()

        # Assert
        temperature = np.linspace(-20, 20, 100) + const.T_tri
        np.testing.assert_allclose(
            Formulae(
                latent_heat_vapourisation="Kirchhoff"
            ).latent_heat_vapourisation.lv(temperature),
            Formulae(latent_heat_vapourisation="Lowe2019").latent_heat_vapourisation.lv(
                temperature
            ),
            rtol=1e-2,
        )

    @staticmethod
    def test_latent_heat_sublimation(plot=False):
        # arrange
        formulae = Formulae()
        T = np.linspace(30, 300) * si.K

        # act
        ls = formulae.latent_heat_sublimation.ls(T)

        # plot
        pyplot.plot(T, ls)
        pyplot.xlabel("T [K]")
        pyplot.ylabel("Lv [J/kg]")
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        np.testing.assert_approx_equal(
            actual=ls[np.abs(T - formulae.trivia.C2K(0)).argmin()],
            desired=2838 * si.kJ / si.kg,
            significant=3,
        )
