"""
Created at 23.04.2020
"""

from PySDM_examples.Arabas_and_Shima_2017_Fig_5.setup import setups
from PySDM_examples.Arabas_and_Shima_2017_Fig_5.simulation import Simulation


def main():
    for setup in setups:
        Simulation(setup).run()


if __name__ == '__main__':
    main()
