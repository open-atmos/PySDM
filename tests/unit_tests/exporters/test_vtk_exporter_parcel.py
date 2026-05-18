# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from unittest.mock import Mock
import numpy as np
import pytest

from PySDM.exporters import VTKExporterParcel


class TestParcelVTKExporter:

    @staticmethod
    @pytest.fixture
    def mock_simulation():
        """Mock simulation object with necessary attributes."""
        simulation = Mock()
        simulation.particulator = Mock()
        simulation.particulator.dt = 1.0
        simulation.particulator.n_sd = 100
        simulation.particulator.environment = Mock()
        simulation.particulator.environment.mass_of_dry_air = 66666

        return simulation

    @staticmethod
    @pytest.fixture
    def mock_output():
        """Mock output data structure."""
        n_levels = 5
        output = {
            "products": {
                "z": np.linspace(0, 100, n_levels),
                "rhod": np.ones(n_levels) * 1.2,
                "S_max_percent": np.linspace(100, 110, n_levels),
            },
            "attributes": {
                "radius": np.random.random((100, 10)),
                "multiplicity": np.random.random((100, 10)),
            },
        }
        return output

    def test_initialization(self, mock_output, mock_simulation):
        # Arrange

        # Act
        exporter = VTKExporterParcel(
            n_sd=mock_simulation.particulator.n_sd,
            output=mock_output,
            mass_of_dry_air=mock_simulation.particulator.environment.mass_of_dry_air,
        )

        # Assert
        assert exporter.output == mock_output
        assert len(exporter.coords["x"]) == mock_simulation.particulator.n_sd
        assert len(exporter.coords["y"]) == mock_simulation.particulator.n_sd
        assert len(exporter.coords["z"]) == mock_simulation.particulator.n_sd
        assert len(exporter.half_diagonal) == exporter.n_levels
        assert exporter.n_levels == len(mock_output["products"]["z"])

    def test_export_products(self, mock_output, mock_simulation, tmp_path):
        # Arrange
        exporter = VTKExporterParcel(
            n_sd=mock_simulation.particulator.n_sd,
            output=mock_output,
            mass_of_dry_air=mock_simulation.particulator.environment.mass_of_dry_air,
        )
        exporter.path = str(tmp_path)
        exporter.attributes_file_path = str(tmp_path / "sd_attributes")
        exporter.products_file_path = str(tmp_path / "sd_products")
        exporter.exported_times = {"products": {}, "attributes": {}}

        step = 1

        # Act
        exporter.export_products(step, mock_simulation)

        # Assert
        expected_file = (
            tmp_path / f"sd_products_num{exporter.add_leading_zeros(step)}.vtu"
        )
        assert expected_file.exists()

        assert len(exporter.exported_times["products"]) == 1
        expected_time = step * mock_simulation.particulator.dt
        assert list(exporter.exported_times["products"].values())[0] == expected_time

    def test_export_attributes(self, mock_output, mock_simulation, tmp_path):
        # Arrange
        exporter = VTKExporterParcel(
            n_sd=mock_simulation.particulator.n_sd,
            output=mock_output,
            mass_of_dry_air=mock_simulation.particulator.environment.mass_of_dry_air,
        )
        exporter.path = str(tmp_path)
        exporter.attributes_file_path = str(tmp_path / "sd_attributes")
        exporter.products_file_path = str(tmp_path / "sd_products")
        exporter.exported_times = {"products": {}, "attributes": {}}

        step = 1

        # Act
        exporter.export_attributes(step, mock_simulation)

        # Assert
        expected_file = (
            tmp_path / f"sd_attributes_num{exporter.add_leading_zeros(step)}.vtu"
        )
        assert expected_file.exists()

        assert len(exporter.exported_times["attributes"]) == 1
        expected_time = step * mock_simulation.particulator.dt
        assert list(exporter.exported_times["attributes"].values())[0] == expected_time

    def test_write_pvd(self, mock_output, mock_simulation, tmp_path):
        # Arrange
        exporter = VTKExporterParcel(
            n_sd=mock_simulation.particulator.n_sd,
            output=mock_output,
            mass_of_dry_air=mock_simulation.particulator.environment.mass_of_dry_air,
        )
        exporter.path = str(tmp_path)
        exporter.attributes_file_path = str(tmp_path / "sd_attributes")
        exporter.products_file_path = str(tmp_path / "sd_products")
        exporter.exported_times = {"products": {}, "attributes": {}}

        steps = [1, 2]
        for step in steps:
            exporter.export_products(step, mock_simulation)
            exporter.export_attributes(step, mock_simulation)

        # Act
        exporter.write_pvd()

        # Assert
        attributes_pvd = tmp_path / "sd_attributes.pvd"
        products_pvd = tmp_path / "sd_products.pvd"

        assert attributes_pvd.exists()
        assert products_pvd.exists()

    def test_coordinate_calculation(self, mock_output, mock_simulation, tmp_path):
        # Arrange
        exporter = VTKExporterParcel(
            n_sd=mock_simulation.particulator.n_sd,
            output=mock_output,
            mass_of_dry_air=mock_simulation.particulator.environment.mass_of_dry_air,
        )
        exporter.path = str(tmp_path)
        exporter.attributes_file_path = str(tmp_path / "sd_attributes")
        exporter.products_file_path = str(tmp_path / "sd_products")
        exporter.exported_times = {"products": {}, "attributes": {}}

        initial_z_coords = exporter.coords["z"].copy()

        # Act
        exporter.export_attributes(step=(step1 := 1), simulation=mock_simulation)
        z_after_step1 = exporter.coords["z"].copy()

        exporter.export_attributes(step=(step2 := 2), simulation=mock_simulation)
        z_after_step2 = exporter.coords["z"].copy()

        # Assert
        delta_z_step1 = (
            mock_output["products"]["z"][step1]
            - mock_output["products"]["z"][step1 - 1]
        )
        expected_z_step1 = initial_z_coords * delta_z_step1
        np.testing.assert_array_equal(z_after_step1, expected_z_step1)

        delta_z_step2 = (
            mock_output["products"]["z"][step2]
            - mock_output["products"]["z"][step2 - 1]
        )
        expected_z_step2 = z_after_step1 + delta_z_step2
        np.testing.assert_array_equal(z_after_step2, expected_z_step2)

    def test_half_diagonal_calculation(self, mock_output, mock_simulation):
        # Arrange

        # Act
        exporter = VTKExporterParcel(
            n_sd=mock_simulation.particulator.n_sd,
            output=mock_output,
            mass_of_dry_air=mock_simulation.particulator.environment.mass_of_dry_air,
        )

        # Assert
        assert len(exporter.half_diagonal) == exporter.n_levels
        assert all(hd > 0 for hd in exporter.half_diagonal)

        volume_0 = (
            mock_simulation.particulator.environment.mass_of_dry_air
            / mock_output["products"]["rhod"][0]
        )
        delta_z_0 = mock_output["products"]["z"][1] - mock_output["products"]["z"][0]
        area_0 = volume_0 / delta_z_0
        expected_hd_0 = (2 * area_0) ** 0.5
        assert abs(exporter.half_diagonal[0] - expected_hd_0) < 1e-10
