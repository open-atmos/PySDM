import numpy as np

from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute
from PySDM.attributes.physics.multiplicities import Multiplicities
from PySDM.attributes.impl.cell_attribute import CellAttribute
from PySDM.state.particles import Particles


class ParticlesFactory:

    @staticmethod
    def attributes(core, req_attr, attributes):
        idx = core.Index.identity_index(core.n_sd)

        extensive_attr = []
        for attr_name in req_attr:
            if isinstance(req_attr[attr_name], ExtensiveAttribute):
                extensive_attr.append(attr_name)
            elif not isinstance(req_attr[attr_name], (DerivedAttribute, Multiplicities, CellAttribute)):
                raise AssertionError()

        extensive_attributes = core.IndexedStorage.empty(idx, (len(extensive_attr), core.n_sd), float)
        for attr in req_attr.values():
            if isinstance(attr, DerivedAttribute):
                attr.allocate(idx)

        extensive_keys = {}
        for i, attr in enumerate(extensive_attr):
            extensive_keys[attr] = i
            req_attr[attr].set_data(extensive_attributes[i, :])
            req_attr[attr].init(attributes[attr])

        n = req_attr['n']
        n.allocate(idx)
        n.init(attributes['n'])
        req_attr['n'].data = core.IndexedStorage.indexed(idx, n.data)
        cell_id = req_attr['cell id']
        cell_id.allocate(idx)
        cell_id.init(attributes['cell id'])
        req_attr['cell id'].data = core.IndexedStorage.indexed(idx, cell_id.data)
        try:
            cell_origin = req_attr['cell origin']
            cell_origin.allocate(idx)
            cell_origin.init(attributes['cell origin'])
            req_attr['cell origin'].data = core.IndexedStorage.indexed(idx, cell_origin.data)
        except KeyError:
            cell_origin = None
        try:
            position_in_cell = req_attr['position in cell']
            position_in_cell.allocate(idx)
            position_in_cell.init(attributes['position in cell'])
            req_attr['position in cell'].data = core.IndexedStorage.indexed(idx, position_in_cell.data)
        except KeyError:
            position_in_cell = None

        cell_start = np.empty(core.mesh.n_cell + 1, dtype=int)

        state = Particles(
            core,
            idx,
            extensive_attributes, extensive_keys,
            cell_start,
            req_attr
        )

        return state

    @staticmethod
    def empty_particles(particles, n_sd) -> Particles:
        idx = particles.Index.identity_index(n_sd)
        return Particles(
            core=particles, idx=idx,
            extensive_attributes=None, extensive_keys={},
            cell_start=np.zeros(0, dtype=np.int64), attributes={}
        )
