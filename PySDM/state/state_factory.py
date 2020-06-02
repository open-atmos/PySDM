"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.state.state import State
from PySDM.attributes.tensive_attribute import TensiveAttribute


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
    def attributes(particles, req_attr, attributes):
        tensive_attr = [attr_name for attr_name in req_attr if isinstance(req_attr[attr_name], TensiveAttribute)]
        extensive_attr = [attr_name for attr_name in tensive_attr if req_attr[attr_name].extensive]
        intensive_attr = [attr_name for attr_name in tensive_attr if not req_attr[attr_name].extensive]
        base_attributes = particles.backend.storage.empty((len(tensive_attr), particles.n_sd), float)  # TODO: divide

        keys = {}
        for i, attr in enumerate(extensive_attr + intensive_attr):
            keys[str(attr)] = i
            req_attr[attr].allocate(base_attributes.read_row(i))
            req_attr[attr].init(attributes[attr])

        n = req_attr['n']
        n.init(attributes['n'])
        cell_id = req_attr['cell id']
        cell_id.init(attributes['cell id'])
        try:
            cell_origin = req_attr['cell origin']
            cell_origin.init(attributes['cell origin'])
        except KeyError:
            cell_origin = None
        try:
            position_in_cell = req_attr['position in cell']
            position_in_cell.init(attributes['position in cell'])
        except KeyError:
            position_in_cell = None

        cell_start = np.empty(particles.mesh.n_cell + 1, dtype=int)

        state = State(n, base_attributes, keys, len(extensive_attr),
                      cell_id, cell_start, cell_origin, position_in_cell, particles, req_attr)
        state.recalculate_cell_id()

        return state

    @staticmethod
    def check_args(n: np.ndarray, intensive: dict, extensive: dict) -> bool:
        assert n.dtype == np.int64 or n.dtype == np.int32

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

