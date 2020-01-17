"""
Created at 09.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.dynamics.condensation import schemes
from PySDM.backends.numba.numba import Numba
import pytest, os
import numpy as np
from matplotlib import pyplot


@pytest.fixture()
def args():
    path = os.path.dirname(os.path.abspath(__file__))

    return np.load(os.path.join(path, "test_data.npy"),
                   allow_pickle=True).item()

# @pytest.mark.skip
class TestBDF:
    # TODO: run for all condensation schemes
    @pytest.mark.parametrize("solver_class", [schemes.BDF, schemes.EE, schemes.ImplicitInSizeExplicitInThermodynamic])
    def test_plot(self, solver_class, args, plot=True):

        ci = args["cell_idx"]

        if plot:
            pyplot.stem(args["v"][ci], args["n"][ci], linefmt='g-')

        def supersaturation(args):
            _, _, RH = Numba.temperature_pressure_RH(rhod=args['rhod'], thd=args['thd'], qv=args['qv'])
            return RH - 1
        sut = solver_class(Numba, 1000)

        S_0 = supersaturation(args)
        sut.step(**args)
        S_1 = supersaturation(args)

        if plot:
            pyplot.stem(args["v"][ci], args["n"][ci], linefmt='b-')
            pyplot.xlabel("particle volume [m^3]")
            pyplot.ylabel("multiplicity")
            pyplot.xlim([2e-14,2.8*10**-14])
            pyplot.legend()
            pyplot.show()

        print(S_0, S_1)