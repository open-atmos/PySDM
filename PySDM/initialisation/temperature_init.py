"""
Crated at 2019
"""

import numpy as np


def temperature_init(ambient_air, cell_id: np.ndarray):
    drop_temperatures = np.empty_like(cell_id)
    for i in range(len(cell_id)):
        drop_temperatures[i] = ambient_air['T'][cell_id[i]]
    return drop_temperatures
