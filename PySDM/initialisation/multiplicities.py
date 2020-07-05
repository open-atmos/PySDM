"""
Created at 16.01.2020
"""

import numpy as np


def n_init(n_per_kg, rhod, mesh, cell_id: np.ndarray):
    n_per_m3 = n_per_kg * rhod[cell_id]
    domain_volume = np.prod(np.array(mesh.size))
    return n_per_m3 * domain_volume


def discretise_n(y_float):
    y_int = y_float.round().astype(np.int64)

    percent_diff = abs(1 - np.sum(y_float) / np.sum(y_int.astype(float)))
    if percent_diff > .01:
        raise Exception(f"{percent_diff}% error in total real-droplet number due to casting multiplicities to ints")

    return y_int
