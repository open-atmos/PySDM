"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.simulation.state.state_factory import StateFactory
from tests.unit_tests.simulation.state.testable_state import TestableState


class TestableStateFactory(StateFactory):

    @staticmethod
    def state(n: np.ndarray, intensive: dict, extensive: dict, positions: (np.ndarray, None),
              particles) -> TestableState:

        assert StateFactory.check_args(n, intensive, extensive)
        sd_num = len(n)
        attributes, keys = StateFactory.init_attributes_and_keys(particles, intensive, extensive, sd_num)

        cell_id, cell_origin, position_in_cell = StateFactory.positions(n, positions)

        state = TestableState(n, attributes, keys, cell_id, cell_origin, position_in_cell, particles)

        state.recalculate_cell_id()
        return state

    @staticmethod
    def state_0d(n: np.ndarray, intensive: dict, extensive: dict, particles) -> TestableState:

        return TestableStateFactory.state(n, intensive, extensive, None, particles)

    @staticmethod
    def state_2d(n: np.ndarray, intensive: dict, extensive: dict, positions: np.ndarray, particles) \
            -> TestableState:

        return TestableStateFactory.state(n, intensive, extensive, positions, particles)

    @staticmethod
    def empty_state(particles) -> TestableState:

        return TestableState(n=np.zeros(0), attributes={}, keys={},
                             cell_id=np.zeros(0), position_in_cell=None, cell_origin=None,
                             particles=particles)
