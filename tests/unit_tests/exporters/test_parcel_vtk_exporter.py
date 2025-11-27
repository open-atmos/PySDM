"""Test module for VTKExporterPyrcel."""

from unittest.mock import Mock
import numpy as np
import pytest

from PySDM.exporters.parcel_vtk_exporter import VTKExporterPyrcel


class TestVTKExporterPyrcel:
    """Test class for VTKExporterPyrcel functionality."""

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
                "n": np.random.random((100, 10)),
            },
        }
        return output

    def test_vtk_exporter_pyrcel_initialization(self, mock_output):
        """Test that VTKExporterPyrcel initializes correctly."""
        # Arrange
        n_sd = 100
        mass_of_dry_air = 66666

        # Act
        exporter = VTKExporterPyrcel(n_sd, mock_output, mass_of_dry_air)

        # Assert
        assert exporter.output == mock_output
        assert len(exporter.x_coords) == n_sd
        assert len(exporter.y_coords) == n_sd
        assert len(exporter.z_coords) == n_sd
        assert len(exporter.half_diagonal) == exporter.n_levels
        assert exporter.n_levels == len(mock_output["products"]["z"])

    def test_vtk_exporter_pyrcel_export_products(
        self, mock_output, mock_simulation, tmp_path
    ):
        """Test exporting products to VTK format."""
        # Arrange
        n_sd = 100
        mass_of_dry_air = 66666

        exporter = VTKExporterPyrcel(n_sd, mock_output, mass_of_dry_air)
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

    def test_vtk_exporter_pyrcel_export_attributes(
        self, mock_output, mock_simulation, tmp_path
    ):
        """Test exporting attributes to VTK format."""
        # Arrange
        n_sd = 100
        mass_of_dry_air = 66666

        exporter = VTKExporterPyrcel(n_sd, mock_output, mass_of_dry_air)
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

    def test_vtk_exporter_pyrcel_write_pvd(
        self, mock_output, mock_simulation, tmp_path
    ):
        """Test writing PVD collection files."""
        # Arrange
        n_sd = 100
        mass_of_dry_air = 66666

        exporter = VTKExporterPyrcel(n_sd, mock_output, mass_of_dry_air)
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

    def test_vtk_exporter_pyrcel_coordinate_calculation(
        self, mock_output, mock_simulation, tmp_path
    ):
        """Test that coordinate calculations work correctly across multiple steps."""
        # Arrange
        exporter = VTKExporterPyrcel(100, mock_output, 666666)
        exporter.path = str(tmp_path)
        exporter.attributes_file_path = str(tmp_path / "sd_attributes")
        exporter.products_file_path = str(tmp_path / "sd_products")
        exporter.exported_times = {"products": {}, "attributes": {}}

        initial_z_coords = exporter.z_coords.copy()

        # Act
        step1 = 1
        exporter.export_attributes(step1, mock_simulation)
        z_after_step1 = exporter.z_coords.copy()

        # Act
        step2 = 2
        exporter.export_attributes(step2, mock_simulation)
        z_after_step2 = exporter.z_coords.copy()

        # Assert
        if step1 != 0:
            delta_z_step1 = (
                mock_output["products"]["z"][step1]
                - mock_output["products"]["z"][step1 - 1]
            )
            expected_z_step1 = initial_z_coords * delta_z_step1
            np.testing.assert_array_equal(z_after_step1, expected_z_step1)

        if step2 != 0:
            delta_z_step2 = (
                mock_output["products"]["z"][step2]
                - mock_output["products"]["z"][step2 - 1]
            )
            expected_z_step2 = z_after_step1 + delta_z_step2
            np.testing.assert_array_equal(z_after_step2, expected_z_step2)

    def test_vtk_exporter_pyrcel_half_diagonal_calculation(self, mock_output):
        """Test the half_diagonal calculation logic."""
        # Arrange
        n_sd = 100
        mass_of_dry_air = 66666

        # Act
        exporter = VTKExporterPyrcel(n_sd, mock_output, mass_of_dry_air)

        # Assert
        assert len(exporter.half_diagonal) == exporter.n_levels
        assert all(hd > 0 for hd in exporter.half_diagonal)

        volume_0 = mass_of_dry_air / mock_output["products"]["rhod"][0]
        delta_z_0 = mock_output["products"]["z"][1] - mock_output["products"]["z"][0]
        area_0 = volume_0 / delta_z_0
        expected_hd_0 = (2 * area_0) ** 0.5
        assert abs(exporter.half_diagonal[0] - expected_hd_0) < 1e-10
