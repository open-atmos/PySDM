"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.state.state_factory import StateFactory
from PySDM_tests.unit_tests.state.testable_state import TestableState


class TestableStateFactory(StateFactory):

    @staticmethod
    def state(n: np.ndarray, intensive: dict, extensive: dict, cell_id, cell_origin, position_in_cell,
              particles) -> TestableState:
        assert StateFactory.check_args(n, intensive, extensive)
        sd_num = len(n)
        attributes, keys, intensive_start = StateFactory.init_attributes_and_keys(particles, intensive, extensive, sd_num)

        if particles.mesh is not None:
            cell_start = np.empty(particles.mesh.n_cell + 1, dtype=int)
        else:
            cell_start = np.empty(0)

        state = TestableState(n, attributes, keys, intensive_start,
                              cell_id, cell_start, cell_origin, position_in_cell, particles)

        state.recalculate_cell_id()
        return state

    @staticmethod
    def state_0d(n: np.ndarray, intensive: dict, extensive: dict, particles) -> TestableState:
        return TestableStateFactory.state(n, intensive, extensive, np.zeros_like(n), None, None, particles)

    @staticmethod
    def state_2d(n: np.ndarray, intensive: dict, extensive: dict, positions: np.ndarray, particles) \
            -> TestableState:
        cell_id, cell_origin, position_in_cell = particles.mesh.cellular_attributes(positions)
        return TestableStateFactory.state(n, intensive, extensive, cell_id, cell_origin, position_in_cell, particles)

    @staticmethod
    def empty_state(particles) -> TestableState:
        return TestableState(n=np.zeros(0), attributes={}, keys={}, intensive_start=-1,
                             cell_id=np.zeros(0, dtype=np.int64), cell_start=np.zeros(0, dtype=np.int64),
                             position_in_cell=None, cell_origin=None,
                             particles=particles)
