"""
factory logic for creating `PySDM.impl.particle_attributes.ParticleAttributes` instances
"""

import numpy as np

from PySDM.attributes.impl import (
    CellAttribute,
    DerivedAttribute,
    DummyAttribute,
    ExtensiveAttribute,
    MaximumAttribute,
)
from PySDM.attributes.physics.multiplicities import Multiplicities
from PySDM.impl.particle_attributes import ParticleAttributes


class ParticleAttributesFactory:
    @staticmethod
    def attributes(particulator, req_attr, attributes):
        # pylint: disable=too-many-locals
        idx = particulator.Index.identity_index(particulator.n_sd)

        extensive_attr = []
        maximum_attr = []
        for attr_name in req_attr:
            if isinstance(req_attr[attr_name], ExtensiveAttribute):
                extensive_attr.append(attr_name)
            elif isinstance(req_attr[attr_name], MaximumAttribute):
                maximum_attr.append(attr_name)
            elif not isinstance(
                req_attr[attr_name],
                (DerivedAttribute, Multiplicities, CellAttribute, DummyAttribute),
            ):
                raise AssertionError()

        extensive_attribute_storage = particulator.IndexedStorage.empty(
            idx, (len(extensive_attr), particulator.n_sd), float
        )
        maximum_attributes = particulator.IndexedStorage.empty(
            idx, (len(maximum_attr), particulator.n_sd), float
        )

        for attr in req_attr.values():
            if isinstance(attr, (DerivedAttribute, DummyAttribute)):
                if attr.name in attributes:
                    raise ValueError(
                        f"attribute '{attr.name}' is a dummy/derived one,"
                        f" but values were provided"
                    )
                attr.allocate(idx)

        extensive_keys = {}
        maximum_keys = {}

        def helper(req_attr, all_attr, names, data, keys):
            for i, attr in enumerate(names):
                keys[attr] = i
                req_attr[attr].set_data(data[i, :])
                try:
                    req_attr[attr].init(all_attr[attr])
                except KeyError as err:
                    raise ValueError(
                        f"attribute '{attr}' requested by one of the components"
                        f" but no initial values given"
                    ) from err

        helper(
            req_attr,
            attributes,
            extensive_attr,
            extensive_attribute_storage,
            extensive_keys,
        )
        helper(req_attr, attributes, maximum_attr, maximum_attributes, maximum_keys)

        n = req_attr["multiplicity"]
        n.allocate(idx)
        n.init(attributes["multiplicity"])
        req_attr["multiplicity"].data = particulator.IndexedStorage.indexed(idx, n.data)
        cell_id = req_attr["cell id"]
        cell_id.allocate(idx)
        cell_id.init(attributes["cell id"])
        req_attr["cell id"].data = particulator.IndexedStorage.indexed(
            idx, cell_id.data
        )
        try:
            cell_origin = req_attr["cell origin"]
            cell_origin.allocate(idx)
            cell_origin.init(attributes["cell origin"])
            req_attr["cell origin"].data = particulator.IndexedStorage.indexed(
                idx, cell_origin.data
            )
        except KeyError:
            cell_origin = None
        try:
            position_in_cell = req_attr["position in cell"]
            position_in_cell.allocate(idx)
            position_in_cell.init(attributes["position in cell"])
            req_attr["position in cell"].data = particulator.IndexedStorage.indexed(
                idx, position_in_cell.data
            )
        except KeyError:
            position_in_cell = None

        cell_start = np.empty(particulator.mesh.n_cell + 1, dtype=int)

        return ParticleAttributes(
            particulator=particulator,
            idx=idx,
            extensive_attribute_storage=extensive_attribute_storage,
            extensive_keys=extensive_keys,
            # maximum_attributes, maximum_keys, # TODO #594
            cell_start=cell_start,
            attributes=req_attr,
        )

    @staticmethod
    def empty_particles(particles, n_sd) -> ParticleAttributes:
        idx = particles.Index.identity_index(n_sd)
        return ParticleAttributes(
            particulator=particles,
            idx=idx,
            extensive_attribute_storage=None,
            extensive_keys={},
            cell_start=np.zeros(2, dtype=np.int64),
            attributes={},
        )
