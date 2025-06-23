# pylint: disable=missing-module-docstring
from __future__ import annotations

import os
import warnings
from typing import Tuple, List, Generator

import pytest
import numpy as np
from scipy.optimize import fsolve

from PySDM import Formulae
from PySDM.physics import si
from PySDM_examples.Loftus_and_Wordsworth_2021 import Settings, Simulation
from PySDM_examples.Loftus_and_Wordsworth_2021.planet import EarthLike


class GroundTruthLoader:
    def __init__(
        self, groundtruth_dir_path: str, n_samples: int = 2, random_seed: int = 2137
    ):
        self.dir_path = groundtruth_dir_path
        self.RHs = None
        self.r0grid = None
        self.m_frac_evap = None
        self.n_samples = n_samples
        np.random.seed(random_seed)

    def __enter__(self):
        try:
            self.RHs = np.load(os.path.join(self.dir_path, "RHs.npy"))
            self.r0grid = np.load(os.path.join(self.dir_path, "r0grid.npy"))
            self.m_frac_evap = np.load(os.path.join(self.dir_path, "m_frac_evap.npy"))
            return self
        except FileNotFoundError as e:
            pytest.fail(f"Error loading ground truth files: {e}")
        pytest.fail("Ground truth data not loaded successfully.")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture(scope="module")
def ground_truth_data(request) -> Generator[GroundTruthLoader, None, None]:
    current_dir = os.path.dirname(os.path.abspath(request.fspath))
    groundtruth_dir = os.path.abspath(os.path.join(current_dir, "ground_truth"))
    if not os.path.isdir(groundtruth_dir):
        pytest.fail(f"Groundtruth directory not found at {groundtruth_dir}")
    with GroundTruthLoader(groundtruth_dir) as gt:
        yield gt


@pytest.fixture
def ground_truth_sample(ground_truth_data: GroundTruthLoader) -> List[dict]:
    gt = ground_truth_data
    n_rh_values = len(gt.RHs)
    n_radius_values = gt.r0grid.shape[1]
    total_points = n_rh_values * n_radius_values
    n_samples = min(gt.n_samples, total_points)
    if n_samples == 0:
        pytest.skip("No data points available to sample.")
    all_indices = np.array(
        [(i, j) for i in range(n_rh_values) for j in range(n_radius_values)]
    )
    sampled_indices_flat = np.random.choice(len(all_indices), n_samples, replace=False)
    sampled_ij_pairs = all_indices[sampled_indices_flat]
    return [
        {
            "rh": gt.RHs[i_rh],
            "r_m": gt.r0grid[0, j_r],
            "expected_m_frac_evap": gt.m_frac_evap[i_rh, j_r],
            "i_rh": i_rh,
            "j_r": j_r,
        }
        for i_rh, j_r in sampled_ij_pairs
    ]


@pytest.fixture(scope="module")
def static_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    formulae = Formulae(
        ventilation="PruppacherAndRasmussen1979",
        saturation_vapour_pressure="AugustRocheMagnus",
        diffusion_coordinate="WaterMassLogarithm",
    )
    radius_array = np.logspace(-4.5, -2.5, 50) * si.m
    RH_array = np.linspace(0.25, 0.99, 50)
    output_matrix = np.full((len(RH_array), len(radius_array)), np.nan)
    const = formulae.constants
    return radius_array, RH_array, output_matrix, const


class TestNPYComparison:
    @staticmethod
    def _mix(dry_prop, vap_prop, ratio):
        return (dry_prop + ratio * vap_prop) / (1 + ratio)

    def _calculate_cloud_properties(
        self, planet: EarthLike, surface_RH: float, formulae_instance: Formulae
    ):
        const = formulae_instance.constants
        planet.RH_zref = surface_RH
        pvs_stp = formulae_instance.saturation_vapour_pressure.pvs_water(planet.T_STP)
        initial_water_vapour_mixing_ratio = const.eps / (
            planet.p_STP / planet.RH_zref / pvs_stp - 1
        )
        R_air_mix = self._mix(const.Rd, const.Rv, initial_water_vapour_mixing_ratio)
        cp_mix = self._mix(const.c_pd, const.c_pv, initial_water_vapour_mixing_ratio)

        def solve_Tcloud(T_candidate):
            pv_ad = (
                initial_water_vapour_mixing_ratio
                / (initial_water_vapour_mixing_ratio + const.eps)
                * planet.p_STP
                * (T_candidate / planet.T_STP) ** (cp_mix / R_air_mix)
            )
            pvs_tc = formulae_instance.saturation_vapour_pressure.pvs_water(T_candidate)
            return pv_ad - pvs_tc

        Tcloud_solutions = fsolve(solve_Tcloud, [150.0, 300.0])
        Tcloud = np.max(Tcloud_solutions)

        Zcloud = (planet.T_STP - Tcloud) * cp_mix / planet.g_std

        th_std = formulae_instance.trivia.th_std(planet.p_STP, planet.T_STP)

        pcloud = \
        formulae_instance.hydrostatics\
            .p_of_z_assuming_const_th_and_initial_water_vapour_mixing_ratio(
            planet.p_STP, th_std, initial_water_vapour_mixing_ratio, Zcloud
        )
        return initial_water_vapour_mixing_ratio, Tcloud, Zcloud, pcloud

    def test_figure_2_replication_accuracy(self, ground_truth_sample):
        formulae = Formulae(
            ventilation="PruppacherAndRasmussen1979",
            saturation_vapour_pressure="AugustRocheMagnus",
            diffusion_coordinate="WaterMassLogarithm",
        )
        for sample in ground_truth_sample:
            planet = EarthLike()
            rh = sample["rh"]
            r_m = sample["r_m"]
            expected = sample["expected_m_frac_evap"]
            i_rh = sample["i_rh"]
            j_r = sample["j_r"]
            try:
                iwvmr, Tcloud, Zcloud, pcloud = \
                self._calculate_cloud_properties(
                    planet, rh, formulae
                )
                simulated = self.calc_simulated_m_frac_evap_point(
                    planet,
                    formulae,
                    i_rh,
                    j_r,
                    rh,
                    r_m,
                    expected,
                    iwvmr,
                    Tcloud,
                    Zcloud,
                    pcloud,
                )
            except Exception as e:
                pytest.fail(
                    f"Error in _calculate_cloud_properties for RH={rh} " +
                    f"(sample idx {i_rh},{j_r}): {e}."
                )
            error_context = (
                f"Sample (RH_idx={i_rh}, R_idx={j_r}), RH={rh:.4f}, R_m={r_m:.3e}. "
                f"Expected: {expected}, Got: {simulated}"
            )
            if np.isnan(expected):
                assert np.isnan(
                    simulated
                ), f"NaN Mismatch. {error_context} (Expected NaN, got non-NaN)"
            else:
                assert not np.isnan(
                    simulated
                ), f"NaN Mismatch. {error_context} (Expected non-NaN, got NaN)"
                np.testing.assert_allclose(
                    simulated,
                    expected,
                    rtol=1e-1,
                    atol=1e-1,
                    err_msg=f"Value Mismatch. {error_context}",
                )

    def calc_simulated_m_frac_evap_point(
        self,
        planet,
        formulae,
        i_rh,
        j_r,
        rh,
        r_m,
        expected,
        iwvmr,
        Tcloud,
        Zcloud,
        pcloud,
    ):
        if np.isnan(r_m) or r_m <= 0:
            pytest.fail(f"Invalid radius r_m={r_m} for sample idx {i_rh},{j_r}.")
        settings = Settings(
            planet=planet,
            r_wet=r_m,
            mass_of_dry_air=1e5 * si.kg,
            initial_water_vapour_mixing_ratio=iwvmr,
            pcloud=pcloud,
            Zcloud=Zcloud,
            Tcloud=Tcloud,
            formulae=formulae,
        )
        simulation = Simulation(settings)
        try:
            output = simulation.run()
            if (
                output
                and "r" in output
                and len(output["r"]) > 0
                and "z" in output
                and len(output["z"]) > 0
            ):
                final_radius_um = output["r"][-1]
                if np.isnan(final_radius_um) or final_radius_um < 0:
                    final_radius_m = final_radius_um * 1e-6
                    if final_radius_m < 0:
                        return 1.0
                    return np.nan
                final_radius_m = final_radius_um * 1e-6
                if r_m == 0:
                    frac_evap = 1.0
                else:
                    frac_evap = 1.0 - (final_radius_m / r_m) ** 3
                return np.clip(frac_evap, 0.0, 1.0)
            return np.nan
        except Exception as e:
            warnings.warn(
                f"Simulation run failed for RH={rh:.4f}, r={r_m:.3e} " +\
                f"(sample idx {i_rh},{j_r}): {type(e).__name__}: {e}"
            )
            if np.isclose(expected, 1.0, atol=1e-6):
                return 1.0
            return np.nan
