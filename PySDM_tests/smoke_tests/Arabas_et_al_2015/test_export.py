from PySDM_examples.Arabas_et_al_2015.netcdf_exporter import NetCDFExporter
from PySDM_examples.Arabas_et_al_2015.settings import Settings
from PySDM_examples.Arabas_et_al_2015.simulation import Simulation
from PySDM_examples.Arabas_et_al_2015.storage import Storage
from PySDM.exporters import VTKExporter
import tempfile

# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


def test_export(backend, tmp_path):
    # Arrange
    settings = Settings()
    settings.simulation_time = settings.dt
    settings.output_interval = settings.dt

    storage = Storage()
    simulator = Simulation(settings, storage, backend)
    _, temp_file = tempfile.mkstemp(dir=tmp_path, suffix='.nc')
    sut = NetCDFExporter(storage, settings, simulator, temp_file)
    
    vtk_exporter = VTKExporter()
    
    simulator.reinit()
    simulator.run(vtk_exporter=vtk_exporter)

    # Act
    sut.run()

    # Assert
