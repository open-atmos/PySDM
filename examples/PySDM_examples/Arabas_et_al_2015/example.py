from open_atmos_jupyter_utils import TemporaryFile
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM_examples.Szumowski_et_al_1998 import Simulation, Storage
from PySDM_examples.utils import DummyController

from PySDM import Formulae
from PySDM.exporters import NetCDFExporter
from PySDM.physics import si


def main():
    settings = Settings(Formulae())

    settings.n_sd_per_gridbox = 25
    settings.grid = (25, 25)
    settings.simulation_time = 5400 * si.second

    storage = Storage()
    simulation = Simulation(settings, storage, SpinUp)
    simulation.reinit()
    simulation.run()
    temp_file = TemporaryFile(".nc")
    exporter = NetCDFExporter(storage, settings, simulation, temp_file.absolute_path)
    exporter.run(controller=DummyController())


if __name__ == "__main__":
    main()
