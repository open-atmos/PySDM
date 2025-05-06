import numpy as np
from matplotlib import pyplot
from matplotlib.pyplot import ylabel

from PySDM_examples.Jouzel_and_Merlivat_1984 import thermodynamic_profiles
from PySDM import Formulae


class TestThermodynamicProfiles:
    @staticmethod
    def test_pressure(plot=True):
        # arrange
        formulae = Formulae()
        T = formulae.trivia.C2K(np.linspace(0, -60))
        p = thermodynamic_profiles.pressure(T)

        # act

        # plot
        pyplot.gca().set(
            ylabel="Pressure [Pa]",
            xlabel="Temperature [K]",
        )
        pyplot.plot(T, p)

        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        # TODO
