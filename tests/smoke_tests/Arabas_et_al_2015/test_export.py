from PySDM.exporters import NetCDFExporter, VTKExporter
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM_examples.Szumowski_et_al_1998 import Simulation, Storage
from PySDM_examples.utils import DummyController
import tempfile
import os

# noinspection PyUnresolvedReferences
from ...backends_fixture import backend


def test_export(backend, tmp_path):
    # Arrange
    settings = Settings()
    settings.simulation_time = settings.dt
    settings.output_interval = settings.dt

    storage = Storage()
    simulator = Simulation(settings, storage, SpinUp=SpinUp, backend=backend)
    _, temp_file = tempfile.mkstemp(dir=tmp_path, suffix='.nc')
    sut = NetCDFExporter(storage, settings, simulator, temp_file)
    
    vtk_exporter = VTKExporter(path=tmp_path)
    
    simulator.reinit()
    simulator.run(vtk_exporter=vtk_exporter)

    vtk_exporter.write_pvd()

    # Act
    sut.run(controller=DummyController())

    # Assert
    filenames_list = os.listdir(os.path.join(tmp_path, 'output'))
    assert len(list(filter(lambda x: x.endswith('.pvd'), filenames_list))) > 0
    assert len(list(filter(lambda x: x.endswith('.vts'), filenames_list))) > 0
    assert len(list(filter(lambda x: x.endswith('.vtu'), filenames_list))) > 0
