import numpy as np

from PySDM.attributes.impl import DerivedAttribute, ExtensiveAttribute, CellAttribute, \
    MaximumAttribute, DummyAttribute
from PySDM.attributes.physics.multiplicities import Multiplicities
from PySDM.state.particles import Particles


class ParticlesFactory:

    @staticmethod
    def attributes(particulator, req_attr, attributes):
        idx = particulator.Index.identity_index(particulator.n_sd)

        extensive_attr = []
        maximum_attr = []
        for attr_name in req_attr:
            if isinstance(req_attr[attr_name], ExtensiveAttribute):
                extensive_attr.append(attr_name)
            elif isinstance(req_attr[attr_name], MaximumAttribute):
                maximum_attr.append(attr_name)
            elif not isinstance(req_attr[attr_name],
                                (DerivedAttribute, Multiplicities, CellAttribute, DummyAttribute)):
                raise AssertionError()

        extensive_attributes = particulator.IndexedStorage.empty(idx, (len(extensive_attr), particulator.n_sd), float)
        maximum_attributes = particulator.IndexedStorage.empty(idx, (len(maximum_attr), particulator.n_sd), float)

        for attr in req_attr.values():
            if isinstance(attr, DerivedAttribute) or isinstance(attr, DummyAttribute):
                attr.allocate(idx)
            if isinstance(attr, DummyAttribute) and attr.name in attributes:
                raise ValueError(f"attribute '{attr.name}' indicated as dummy but values were provided")

        extensive_keys = {}
        maximum_keys = {}

        def helper(req_attr, all_attr, names, data, keys):
            for i, attr in enumerate(names):
                keys[attr] = i
                req_attr[attr].set_data(data[i, :])
                try:
                    req_attr[attr].init(all_attr[attr])
                except KeyError:
                    raise ValueError(f"attribute '{attr}' required by one of the dynamics but no initial values given")

        helper(req_attr, attributes, extensive_attr, extensive_attributes, extensive_keys)
        helper(req_attr, attributes, maximum_attr, maximum_attributes, maximum_keys)

        n = req_attr['n']
        n.allocate(idx)
        n.init(attributes['n'])
        req_attr['n'].data = particulator.IndexedStorage.indexed(idx, n.data)
        cell_id = req_attr['cell id']
        cell_id.allocate(idx)
        cell_id.init(attributes['cell id'])
        req_attr['cell id'].data = particulator.IndexedStorage.indexed(idx, cell_id.data)
        try:
            cell_origin = req_attr['cell origin']
            cell_origin.allocate(idx)
            cell_origin.init(attributes['cell origin'])
            req_attr['cell origin'].data = particulator.IndexedStorage.indexed(idx, cell_origin.data)
        except KeyError:
            cell_origin = None
        try:
            position_in_cell = req_attr['position in cell']
            position_in_cell.allocate(idx)
            position_in_cell.init(attributes['position in cell'])
            req_attr['position in cell'].data = particulator.IndexedStorage.indexed(idx, position_in_cell.data)
        except KeyError:
            position_in_cell = None

        cell_start = np.empty(particulator.mesh.n_cell + 1, dtype=int)

        state = Particles(
            particulator,
            idx,
            extensive_attributes, extensive_keys,
            # maximum_attributes, maximum_keys, # TODO #594
            cell_start,
            req_attr
        )

        return state

    @staticmethod
    def empty_particles(particles, n_sd) -> Particles:
        idx = particles.Index.identity_index(n_sd)
        return Particles(
            particulator=particles, idx=idx,
            extensive_attributes=None, extensive_keys={},
            cell_start=np.zeros(0, dtype=np.int64), attributes={}
        )
