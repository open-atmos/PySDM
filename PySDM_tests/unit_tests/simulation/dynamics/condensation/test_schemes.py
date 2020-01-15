"""
Created at 09.01.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.dynamics.condensation import schemes
import pytest, os
import numpy as np
from matplotlib import pyplot


@pytest.fixture(scope='module')
def args():
    path = os.path.dirname(os.path.abspath(__file__))

    return np.load(os.path.join(path, "test_data.npy"),
                   allow_pickle=True).item()


class TestBDF:
    def test_plot(self, args, plot=False):

        ci = args["cell_idx"]

        if plot:
            pyplot.stem(args["v"][ci], args["n"][ci], linefmt='g-')

        schemes.BDF.step(**args)

        if plot:
            pyplot.stem(args["v"][ci], args["n"][ci], linefmt='b-')
            pyplot.legend()
            pyplot.show()
