from PySDM_examples.Arabas_and_Shima_2017.settings import setups
from PySDM_examples.Arabas_and_Shima_2017.simulation import Simulation


def main():
    for settings in setups:
        Simulation(settings).run()


if __name__ == "__main__":
    main()
