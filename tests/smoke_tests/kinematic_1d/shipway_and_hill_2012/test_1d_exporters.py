# pylint: disable = missing-module-docstring,missing-class-docstring,missing-function-docstring
import os
import platform
from collections import namedtuple
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from open_atmos_jupyter_utils import TemporaryFile
from PySDM_examples.Shipway_and_Hill_2012 import Settings, Simulation
from PySDM_examples.utils import readVTK_1d

from PySDM.exporters import NetCDFExporter_1d, VTKExporter_1d, readNetCDF_1d
from PySDM.physics import si


@pytest.fixture(name="simulation_1d")
def simulation_1d_fixture():
    n_sd_per_gridbox = 16
    settings = Settings(
        n_sd_per_gridbox=n_sd_per_gridbox,
        dt=60 * si.s,
        dz=200 * si.m,
        precip=True,
        rho_times_w_1=2 * si.m / si.s * si.kg / si.m**3,
    )
    settings.t_max = 20 * settings.dt
    settings.save_spec_and_attr_times = [0 * si.min, 10 * si.min, 20 * si.min]
    simulation = Simulation(settings)
    results = simulation.run()
    return namedtuple(
        Path(__file__).stem + "_FixtureData", ("results, settings, simulation")
    )(results=results, settings=settings, simulation=simulation)


class Test2DExporters:
    @staticmethod
    @pytest.mark.parametrize(
        "exclude_particle_reservoir",
        (
            False,
            True,
        ),
    )
    def test_netcdf_exporter_1d(simulation_1d, exclude_particle_reservoir):
        # Arrange
        data = simulation_1d.results.products
        settings = simulation_1d.settings
        nz_export = (
            int(settings.z_max / settings.dz)
            if exclude_particle_reservoir
            else settings.nz
        )

        # Act
        file = TemporaryFile(".nc")
        netcdf_exporter = NetCDFExporter_1d(
            data,
            settings,
            simulation_1d.simulation,
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
        assert np.amin(data_from_file.products["height"]) == np.amin(
            data["z"][-nz_export:]
        )
        assert np.amax(data_from_file.products["height"]) == np.amax(
            data["z"][-nz_export:]
        )
        assert data_from_file.products["rhod"].mean() == pytest.approx(
            data["rhod"][-nz_export:, :].mean(), 1e-6
        )
        assert data_from_file.products["wet spectrum"].mean() == pytest.approx(
            data["wet spectrum"][-nz_export:, :, :].mean(), 1e-6
        )

        assert data_from_file.settings["precip"] == settings.precip
        assert data_from_file.settings["kappa"] == settings.kappa
        assert (
            data_from_file.settings["r_bins_edges"].size == settings.number_of_bins + 1
        )

    @staticmethod
    @pytest.mark.skipif(
        platform.architecture()[0] == "32bit", reason="Not available vtk module!"
    )
    @pytest.mark.parametrize(
        "exclude_particle_reservoir",
        (
            False,
            True,
        ),
    )
    def test_vtk_exporter_1d(
        simulation_1d, exclude_particle_reservoir
    ):  # pylint: disable=too-many-locals
        # Arrange
        data = simulation_1d.results.attributes
        settings = simulation_1d[1]
        z0 = 0.0 if exclude_particle_reservoir else -settings.particle_reservoir_depth
        exported_particles_indexes = {}
        number_of_exported_particles = []
        for i, t in enumerate(settings.save_spec_and_attr_times):
            exported_particles_indexes[t] = np.where(
                data["cell origin"][i][0]
                >= int((z0 + settings.particle_reservoir_depth) / settings.dz)
            )
            number_of_exported_particles.append(exported_particles_indexes[t][0].size)

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
            data_from_file = {}
            for t in settings.save_spec_and_attr_times:
                leading_zeros_in_filename = [
                    "0" for i in range(len(str(settings.t_max)) - len(str(t)))
                ]
                filename = "time" + "".join(leading_zeros_in_filename) + str(t) + ".vtu"
                data_from_file[t] = readVTK_1d(tmpdir + filename)

        # Assert
        for i, t in enumerate(settings.save_spec_and_attr_times):
            filename_leading_zeros = "".join(
                ["0" for i in range(len(str(settings.t_max)) - len(str(t)))]
            )
            filename = "time" + filename_leading_zeros + str(t) + ".vtu"
            assert filename in written_files_list
            assert z0 <= np.amin(data_from_file[t]["z"])
            assert np.amax(data_from_file[t]["z"]) <= settings.z_max
            assert data_from_file[t]["z"].size == number_of_exported_particles[i]
            data_from_file[t].pop("z")
            assert data.keys() == data_from_file[t].keys()
            assert (
                data["radius"][i][exported_particles_indexes[t]].mean()
                == data_from_file[t]["radius"].mean()
            )
