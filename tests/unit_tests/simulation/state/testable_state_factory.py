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
    def state(n: np.ndarray, grid: tuple, intensive: dict, extensive: dict, positions: (np.ndarray, None),
              backend) -> TestableState:

        assert StateFactory.check_args(n, intensive, extensive)
        sd_num = len(n)
        attributes, keys = StateFactory.init_attributes_and_keys(backend, intensive, extensive, sd_num)

        cell_id, cell_origin, position_in_cell = StateFactory.positions(n, positions)

        state = TestableState(n, grid, attributes, keys, cell_id, cell_origin, position_in_cell, backend)

        state.recalculate_cell_id()
        return state

    @staticmethod
    def state_0d(n: np.ndarray, intensive: dict, extensive: dict, backend) -> TestableState:

        return TestableStateFactory.state(n, (), intensive, extensive, None, backend)

    @staticmethod
    def state_2d(n: np.ndarray, grid: tuple, intensive: dict, extensive: dict, positions: np.ndarray, backend) \
            -> TestableState:

        return TestableStateFactory.state(n, grid, intensive, extensive, positions, backend)

    @staticmethod
    def empty_state(backend) -> TestableState:

        return TestableState(n=np.zeros(0), grid=(), attributes={}, keys={},
                             cell_id=np.zeros(0), position_in_cell=None, cell_origin=None,
                             backend=backend)
