"""
Created at 09.11.2019
"""

import numpy as np

from PySDM.attributes.tensive_attribute import TensiveAttribute
from PySDM.state.particles import Particles


class ParticlesFactory:

    @staticmethod
    def attributes(core, req_attr, attributes):
        tensive_attr = [attr_name for attr_name in req_attr if isinstance(req_attr[attr_name], TensiveAttribute)]
        extensive_attr = [attr_name for attr_name in tensive_attr if req_attr[attr_name].extensive]
        intensive_attr = [attr_name for attr_name in tensive_attr if not req_attr[attr_name].extensive]
        idx = core.backend.Index.from_ndarray(np.arange(core.n_sd))
        base_attributes = core.backend.IndexedStorage.empty((len(tensive_attr), core.n_sd), float)  # TODO: divide
        base_attributes = core.backend.IndexedStorage.indexed(idx, base_attributes)

        keys = {}
        for i, attr in enumerate(extensive_attr + intensive_attr):
            keys[str(attr)] = i
            req_attr[attr].allocate(base_attributes.read_row(i))
            req_attr[attr].init(attributes[attr])

        n = req_attr['n']
        n.init(attributes['n'])
        req_attr['n'].data = core.backend.IndexedStorage.indexed(idx, n.data)
        cell_id = req_attr['cell id']
        cell_id.init(attributes['cell id'])
        req_attr['cell id'].data = core.backend.IndexedStorage.indexed(idx, cell_id.data)
        try:
            cell_origin = req_attr['cell origin']
            cell_origin.init(attributes['cell origin'])
            req_attr['cell origin'].data = core.backend.IndexedStorage.indexed(idx, cell_origin.data)
        except KeyError:
            cell_origin = None
        try:
            position_in_cell = req_attr['position in cell']
            position_in_cell.init(attributes['position in cell'])
            req_attr['position in cell'].data = core.backend.IndexedStorage.indexed(idx, position_in_cell.data)
        except KeyError:
            position_in_cell = None

        cell_start = np.empty(core.mesh.n_cell + 1, dtype=int)

        state = Particles(core, idx, base_attributes, keys, len(extensive_attr), cell_start, req_attr)

        return state

    @staticmethod
    def empty_particles(particles, n_sd) -> Particles:
        idx = particles.backend.Index.from_ndarray(np.arange(n_sd))
        return Particles(
            core=particles, idx=idx, keys={}, intensive_start=-1,
            cell_start=np.zeros(0, dtype=np.int64), base_attributes=None, attributes={})
