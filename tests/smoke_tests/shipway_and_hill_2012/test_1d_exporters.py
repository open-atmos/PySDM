# pylint: disable = missing-module-docstring,missing-class-docstring,missing-function-docstring,redefined-outer-name
import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from atmos_cloud_sim_uj_utils import TemporaryFile
from PySDM_examples.Shipway_and_Hill_2012 import Settings, Simulation

from PySDM.exporters import NetCDFExporter_1d, VTKExporter_1d, readNetCDF_1d
from PySDM.physics import si


@pytest.fixture
def simulation_1d():
    n_sd_per_gridbox = 16
    settings = Settings(
        n_sd_per_gridbox=n_sd_per_gridbox,
        dt=60 * si.s,
        dz=200 * si.m,
        precip=False,
        rho_times_w_1=2 * si.m / si.s * si.kg / si.m**3,
    )
    settings.t_max = 20 * settings.dt
    settings.save_spec_and_attr_times = [0 * si.min, 10 * si.min, 20 * si.min]
    simulation = Simulation(settings)
    results = simulation.run()
    return results, settings, simulation


@pytest.mark.parametrize(
    "exclude_particle_reservoir",
    (
        False,
        True,
    ),
)
def test_netcdf_exporter_1d(simulation_1d, exclude_particle_reservoir):
    # Arrange
    data = simulation_1d[0].products
    settings = simulation_1d[1]
    simulation = simulation_1d[2]
    nz_export = (
        int(settings.z_max / settings.dz) if exclude_particle_reservoir else settings.nz
    )

    # Act
    file = TemporaryFile(".nc")
    netcdf_exporter = NetCDFExporter_1d(
        data,
        settings,
        simulation,
        filename=file.absolute_path,
        exclude_particle_reservoir=exclude_particle_reservoir,
    )
    netcdf_exporter.run()
    data_from_file = readNetCDF_1d(file.absolute_path)

    # Assert
    assert data_from_file.products["time"].shape == data["t"].shape
    assert data_from_file.products["height"].shape == data["z"][-nz_export:].shape
    assert data_from_file.products["T"].shape == data["T"][-nz_export:, :].shape
    assert (
        data_from_file.products["dry spectrum"].shape
        == data["dry spectrum"][-nz_export:, :, :].shape
    )

    assert np.amin(data_from_file.products["time"]) == np.amin(data["t"])
    assert np.amax(data_from_file.products["time"]) == np.amax(data["t"])
    assert np.amin(data_from_file.products["height"]) == np.amin(data["z"][-nz_export:])
    assert np.amax(data_from_file.products["height"]) == np.amax(data["z"][-nz_export:])
    assert data_from_file.products["rhod"].mean() == pytest.approx(
        data["rhod"][-nz_export:, :].mean(), 1e-6
    )
    assert data_from_file.products["wet spectrum"].mean() == pytest.approx(
        data["wet spectrum"][-nz_export:, :, :].mean(), 1e-6
    )

    assert data_from_file.settings["precip"] == settings.precip
    assert data_from_file.settings["kappa"] == settings.kappa
    assert data_from_file.settings["r_bins_edges"].size == settings.number_of_bins + 1


@pytest.mark.parametrize(
    "exclude_particle_reservoir",
    (
        False,
        True,
    ),
)
def test_vtk_exporter_1d(simulation_1d, exclude_particle_reservoir):
    # Arrange
    data = simulation_1d[0].attributes
    settings = simulation_1d[1]

    # Act
    with TemporaryDirectory() as tmpdir_:
        tmpdir = tmpdir_ + "/"
        vtk_exporter = VTKExporter_1d(
            data,
            settings,
            path=tmpdir,
            exclude_particle_reservoir=exclude_particle_reservoir,
        )
        vtk_exporter.run()
        written_files_list = os.listdir(tmpdir)

    # Assert
    for t in settings.save_spec_and_attr_times:
        filename_leading_zeros = "".join(
            ["0" for i in range(len(str(settings.t_max)) - len(str(t)))]
        )
        filename = "time" + filename_leading_zeros + str(t) + ".vtu"
        assert filename in written_files_list
