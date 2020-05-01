"""
Created at 25.11.2019

@author: Michael Olesik
@author: Sylwester Arabas
"""

from PySDM_examples.Yang_et_al_2018_Fig_2.setup import Setup
from PySDM_examples.Yang_et_al_2018_Fig_2.simulation import Simulation


if __name__ == '__main__':
    Simulation(setup=Setup()).run()
