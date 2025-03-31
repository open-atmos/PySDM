"""
collision kernel from Hall et. al. 1980.
"""

import numpy as np
import numba as nb
from scipy.interpolate import RegularGridInterpolator
from PySDM.dynamics.collisions.collision_kernels.geometric import Geometric


class Hall(Geometric):

    def table(self, collector_radius_m, collected_radius_m):

        collector_radii_um = np.array(
            [300, 200, 150, 100, 70, 60, 50, 40, 30, 20, 10], dtype=float
        )
        ratio_values = np.array(
            [
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                1.00,
            ],
            dtype=float,
        )

        table_data = np.array(
            [
                # For collector radius = 300 µm:
                [
                    0.97,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                # For collector radius = 200 µm:
                [
                    0.87,
                    0.96,
                    0.98,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                # For collector radius = 150 µm:
                [
                    0.77,
                    0.93,
                    0.97,
                    0.97,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                # For collector radius = 100 µm:
                [
                    0.50,
                    0.79,
                    0.91,
                    0.95,
                    0.95,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                # For collector radius = 70 µm:
                [
                    0.20,
                    0.58,
                    0.75,
                    0.84,
                    0.88,
                    0.90,
                    0.92,
                    0.94,
                    0.95,
                    0.95,
                    0.95,
                    0.95,
                    0.95,
                    0.95,
                    0.97,
                    1.0,
                    1.02,
                    1.04,
                    2.3,
                    4.0,
                ],
                # For collector radius = 60 µm:
                [
                    0.05,
                    0.43,
                    0.64,
                    0.77,
                    0.84,
                    0.87,
                    0.89,
                    0.90,
                    0.91,
                    0.91,
                    0.91,
                    0.91,
                    0.91,
                    0.92,
                    0.93,
                    0.95,
                    1.0,
                    1.03,
                    1.7,
                    3.0,
                ],
                # For collector radius = 50 µm:
                [
                    0.005,
                    0.40,
                    0.60,
                    0.70,
                    0.78,
                    0.83,
                    0.86,
                    0.88,
                    0.90,
                    0.90,
                    0.90,
                    0.90,
                    0.89,
                    0.88,
                    0.88,
                    0.89,
                    0.92,
                    1.01,
                    1.3,
                    2.3,
                ],
                # For collector radius = 40 µm:
                [
                    0.001,
                    0.07,
                    0.28,
                    0.50,
                    0.62,
                    0.68,
                    0.74,
                    0.78,
                    0.80,
                    0.80,
                    0.80,
                    0.78,
                    0.77,
                    0.76,
                    0.77,
                    0.77,
                    0.78,
                    0.79,
                    0.95,
                    1.4,
                ],
                # For collector radius = 30 µm:
                [
                    0.001,
                    0.002,
                    0.02,
                    0.04,
                    0.085,
                    0.17,
                    0.27,
                    0.40,
                    0.50,
                    0.55,
                    0.58,
                    0.59,
                    0.58,
                    0.54,
                    0.51,
                    0.49,
                    0.47,
                    0.45,
                    0.47,
                    0.52,
                ],
                # For collector radius = 20 µm:
                [
                    0.0001,
                    0.0001,
                    0.005,
                    0.016,
                    0.022,
                    0.03,
                    0.043,
                    0.052,
                    0.064,
                    0.072,
                    0.079,
                    0.082,
                    0.080,
                    0.076,
                    0.067,
                    0.057,
                    0.048,
                    0.040,
                    0.033,
                    0.027,
                ],
                # For collector radius = 10 µm:
                [
                    0.0001,
                    0.0001,
                    0.0001,
                    0.014,
                    0.017,
                    0.019,
                    0.022,
                    0.027,
                    0.030,
                    0.033,
                    0.035,
                    0.037,
                    0.038,
                    0.038,
                    0.037,
                    0.036,
                    0.035,
                    0.032,
                    0.029,
                    0.027,
                ],
            ],
            dtype=float,
        )

        # Sort collector radii ascending:
        idx_sorted = np.argsort(collector_radii_um)
        collector_radii_um_sorted = collector_radii_um[idx_sorted]
        table_data_sorted = table_data[idx_sorted, :]

        interpolator = RegularGridInterpolator(
            (collector_radii_um_sorted, ratio_values),
            table_data_sorted,
            bounds_error=False,
            fill_value=None,
        )

        R_um = collector_radius_m * 1e6
        r_um = collected_radius_m * 1e6

        ratio = r_um / R_um

        if ratio > 1:
            ratio = 1 / ratio
            R_um, r_um = r_um, R_um

        result = interpolator([R_um, ratio])

        return float(result)

    @nb.njit(parallel=True)
    def __call__(self, output, is_first_in_pair):
        output.sum(self.particulator.attributes["radius"], is_first_in_pair)
        output **= 2
        output *= np.pi
        self.pair_tmp.distance(
            self.particulator.attributes["relative fall velocity"], is_first_in_pair
        )
        output *= self.pair_tmp
        idx = self.particulator.attributes["radius"].idx
        for i in nb.prange(len(idx) - 1):
            if is_first_in_pair.indicator.data[i]:
                temp = float(output[i // 2])
                temp *= self.table(
                    self.particulator.attributes["radius"].data[idx[i]],
                    self.particulator.attributes["radius"].data[idx[i + 1]],
                )
                output[i // 2] = temp
