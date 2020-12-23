"""
Created at 23.04.2020
"""

from PySDM_examples.Arabas_and_Shima_2017_Fig_5.settings import setups
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.simulation import Simulation


def main():
    for settings in setups:
        Simulation(settings).run()


if __name__ == '__main__':
    main()
