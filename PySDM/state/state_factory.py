"""
Created at 09.11.2019
"""

import numpy as np
from PySDM.state.state import State
from PySDM.attributes.tensive_attribute import TensiveAttribute


class StateFactory:

    @staticmethod
    def attributes(particles, req_attr, attributes):
        tensive_attr = [attr_name for attr_name in req_attr if isinstance(req_attr[attr_name], TensiveAttribute)]
        extensive_attr = [attr_name for attr_name in tensive_attr if req_attr[attr_name].extensive]
        intensive_attr = [attr_name for attr_name in tensive_attr if not req_attr[attr_name].extensive]
        idx = particles.backend.IndexedStorage.from_ndarray(np.arange(particles.n_sd))
        base_attributes = particles.backend.IndexedStorage.empty((len(tensive_attr), particles.n_sd), float)  # TODO: divide
        base_attributes = particles.backend.IndexedStorage.indexed(idx, base_attributes)

        keys = {}
        for i, attr in enumerate(extensive_attr + intensive_attr):
            keys[str(attr)] = i
            req_attr[attr].allocate(base_attributes.read_row(i))
            req_attr[attr].init(attributes[attr])

        n = req_attr['n']
        n.init(attributes['n'])
        req_attr['n'].data = particles.backend.IndexedStorage.indexed(idx, n.data)
        cell_id = req_attr['cell id']
        cell_id.init(attributes['cell id'])
        req_attr['cell id'].data = particles.backend.IndexedStorage.indexed(idx, cell_id.data)
        try:
            cell_origin = req_attr['cell origin']
            cell_origin.init(attributes['cell origin'])
            req_attr['cell origin'].data = particles.backend.IndexedStorage.indexed(idx, cell_origin.data)
        except KeyError:
            cell_origin = None
        try:
            position_in_cell = req_attr['position in cell']
            position_in_cell.init(attributes['position in cell'])
            req_attr['position in cell'].data = particles.backend.IndexedStorage.indexed(idx, position_in_cell.data)
        except KeyError:
            position_in_cell = None

        cell_start = np.empty(particles.mesh.n_cell + 1, dtype=int)

        state = State(particles, idx, base_attributes, keys, len(extensive_attr), cell_start, req_attr)

        return state

    @staticmethod
    def empty_state(particles, n_sd) -> State:
        idx = particles.backend.IndexedStorage.from_ndarray(np.arange(n_sd))
        return State(
            core=particles, idx=idx, keys={}, intensive_start=-1,
            cell_start=np.zeros(0, dtype=np.int64), base_attributes=None, attributes={})
