"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.simulation.state.state import State


class StateFactory:

    @staticmethod
    def state(n: np.ndarray, intensive: dict, extensive: dict, cell_id, cell_origin, position_in_cell,
              particles) -> State:
        assert StateFactory.check_args(n, intensive, extensive)
        sd_num = len(n)
        attributes, keys, intensive_start = StateFactory.init_attributes_and_keys(
            particles, intensive, extensive, sd_num)

        cell_start = np.empty(particles.mesh.n_cell + 1, dtype=int)
        state = State(n, attributes, keys, intensive_start,
                      cell_id, cell_start, cell_origin, position_in_cell, particles)
        state.recalculate_cell_id()

        return state

    @staticmethod
    def check_args(n: np.ndarray, intensive: dict, extensive: dict) -> bool:
        result = True
        if n.ndim != 1:
            result = False

        # https://en.wikipedia.org/wiki/Intensive_and_extensive_properties
        for attribute in (*intensive.values(), *extensive.values()):
            if attribute.shape != n.shape:
                result = False
                break
        return result

    @staticmethod
    def init_attributes_and_keys(particles, intensive: dict, extensive: dict, SD_num) -> (np.ndarray, dict, int):
        attributes = particles.backend.array((len(intensive) + len(extensive), SD_num), float)
        keys = {}
        intensive_start = len(extensive)

        idx = 0
        for tensive in ['extensive', 'intensive']:
            for key, array in {'intensive': intensive, 'extensive': extensive}[tensive].items():
                if key in keys:
                    raise ValueError("Non-unique attribute name: " + key)
                keys[key] = idx
                particles.backend.write_row(attributes, idx, particles.backend.from_ndarray(array))
                idx += 1

        return attributes, keys, intensive_start

