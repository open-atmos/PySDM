# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os
import tempfile
from tempfile import TemporaryDirectory

from open_atmos_jupyter_utils import TemporaryFile
from PySDM_examples.Arabas_et_al_2015 import Settings, SpinUp
from PySDM_examples.Szumowski_et_al_1998.gui_settings import GUISettings
from PySDM_examples.Szumowski_et_al_1998.simulation import Simulation
from PySDM_examples.Szumowski_et_al_1998.storage import Storage
from PySDM_examples.utils import DummyController
from PySDM_examples.utils.widgets import IntSlider
from scipy.io import netcdf_file

from PySDM import Formulae
from PySDM.backends import CPU
from PySDM.exporters import NetCDFExporter, VTKExporter


def test_export(backend_class, tmp_path):
    # Arrange
    settings = Settings()
    settings.simulation_time = settings.dt
    settings.output_interval = settings.dt

    storage = Storage()
    simulator = Simulation(
        settings, storage, SpinUp=SpinUp, backend_class=backend_class
    )
    _, temp_file = tempfile.mkstemp(dir=tmp_path, suffix=".nc")
    sut = NetCDFExporter(storage, settings, simulator, temp_file)

    vtk_exporter = VTKExporter(path=tmp_path)

    simulator.reinit()
    simulator.run(vtk_exporter=vtk_exporter)

    vtk_exporter.write_pvd()

    # Act
    sut.run(controller=DummyController())

    # Assert
    filenames_list = os.listdir(os.path.join(tmp_path, "output"))
    assert len(list(filter(lambda x: x.endswith(".pvd"), filenames_list))) > 0
    assert len(list(filter(lambda x: x.endswith(".vts"), filenames_list))) > 0
    assert len(list(filter(lambda x: x.endswith(".vtu"), filenames_list))) > 0


def test_export_with_gui_settings():
    # Arrange
    settings = GUISettings(Settings(Formulae()))
    settings.ui_nz.value += 1
    settings.ui_simulation_time = IntSlider(value=10)
    settings.ui_dt = IntSlider(value=10)
    settings.ui_output_options["interval"] = IntSlider(value=settings.ui_dt.value)
    assert settings.n_steps == 1
    assert len(settings.output_steps) == 2 and settings.output_steps[-1] == 1

    storage = Storage()
    simulator = Simulation(
        settings=settings, storage=storage, SpinUp=SpinUp, backend_class=CPU
    )
    file = TemporaryFile()
    ncdf_exporter = NetCDFExporter(
        storage=storage,
        settings=settings,
        simulator=simulator,
        filename=file.absolute_path,
    )
    with TemporaryDirectory() as tempdir:
        vtk_exporter = VTKExporter(path=tempdir)

        # Act
        simulator.reinit()
        simulator.run(vtk_exporter=vtk_exporter)
        ncdf_exporter.run(controller=DummyController())
        vtk_exporter.write_pvd()

        # Assert
        versions = netcdf_file(file.absolute_path).versions  # pylint: disable=no-member
        assert "PyMPDATA" in str(versions)

        filenames_list = os.listdir(os.path.join(tempdir, "output"))
    assert len(list(filter(lambda x: x.endswith(".pvd"), filenames_list))) == 2
    assert len(list(filter(lambda x: x.endswith(".vts"), filenames_list))) == 2
    assert len(list(filter(lambda x: x.endswith(".vtu"), filenames_list))) == 2
