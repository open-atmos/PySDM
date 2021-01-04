"""
Created at 25.11.2019
"""

from PySDM_examples.Yang_et_al_2018_Fig_2.settings import Settings
from PySDM_examples.Yang_et_al_2018_Fig_2.simulation import Simulation


if __name__ == '__main__':
    Simulation(settings=Settings()).run()
