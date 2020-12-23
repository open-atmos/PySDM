"""
Created at 23.10.2019
"""

import numpy as np
from .displacement_settings import DisplacementSettings
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


class TestExplicitEulerWithInterpolation:

    @staticmethod
    def test_single_cell(backend):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO!!!

        # Arrange
        settings = DisplacementSettings()
        settings.courant_field_data = (np.array([[.1, .2]]).T, np.array([[.3, .4]]))
        settings.positions = [[0.5], [0.5]]
        sut, _ = settings.get_displacement(backend)

        # Act
        sut()

        # Assert
        pass

    @staticmethod
    def test_advection(backend):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO!!!

        # Arrange
        settings = DisplacementSettings()
        settings.grid = (3, 3)
        settings.courant_field_data = (np.ones((4, 3)), np.zeros((3, 4)))
        settings.positions = [[1.5], [1.5]]
        sut, core = settings.get_displacement(backend)

        # Act
        sut()

        # Assert
        np.testing.assert_array_equal(core.particles['cell origin'][:, 0], np.array([2, 1]))

    @staticmethod
    def test_calculate_displacement(backend):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO!!!

        # Arrange
        settings = DisplacementSettings()
        a = .1
        b = .2
        w = .25
        settings.courant_field_data = (np.array([[a, b]]).T, np.array([[0, 0]]))
        settings.positions = [[w], [0]]
        settings.scheme = 'FTFS'
        sut, core = settings.get_displacement(backend)

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant,
                                   core.particles['cell origin'], core.particles['position in cell'])

        # Assert
        np.testing.assert_equal(sut.displacement[0, 0], (1 - w) * a + w * b)

    @staticmethod
    def test_calculate_displacement_dim1(backend):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO!!!

        # Arrange
        settings = DisplacementSettings()
        a = .1
        b = .2
        w = .25
        settings.courant_field_data = (np.array([[0, 0]]).T, np.array([[a, b]]))
        settings.positions = [[0], [w]]
        settings.scheme = 'FTFS'
        sut, core = settings.get_displacement(backend)

        # Act
        sut.calculate_displacement(sut.displacement, sut.courant,
                                   core.particles['cell origin'], core.particles['position in cell'])

        # Assert
        np.testing.assert_equal(sut.displacement[1, 0], (1 - w) * a + w * b)

    @staticmethod
    def test_update_position(backend):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO!!!

        # Arrange
        settings = DisplacementSettings()
        px = .1
        py = .2
        settings.positions = [[px], [py]]
        sut, core = settings.get_displacement(backend)

        droplet_id = 0
        sut.displacement[0, droplet_id] = .1
        sut.displacement[1, droplet_id] = .2

        # Act
        sut.update_position(core.particles['position in cell'], sut.displacement)

        # Assert
        for d in range(2):
            assert core.particles['position in cell'][d, droplet_id] == (
                    settings.positions[d][droplet_id] + sut.displacement[d, droplet_id]
            )

    @staticmethod
    def test_update_cell_origin(backend):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO!!!

        # Arrange
        settings = DisplacementSettings()
        sut, core = settings.get_displacement(backend)

        droplet_id = 0
        state = core.particles
        state['position in cell'][0, droplet_id] = 1.1
        state['position in cell'][1, droplet_id] = 1.2

        # Act
        sut.update_cell_origin(state['cell origin'], state['position in cell'])

        # Assert
        for d in range(2):
            assert state['cell origin'][d, droplet_id] == settings.positions[d][droplet_id] + 1
            assert state['position in cell'][d, droplet_id] == (state['position in cell'][d, droplet_id]
                                                                - np.floor(state['position in cell'][d, droplet_id]))

    @staticmethod
    def test_boundary_condition(backend):
        from PySDM.backends import ThrustRTC
        if backend is ThrustRTC:
            return  # TODO!!!

        # Arrange
        settings = DisplacementSettings()
        sut, core = settings.get_displacement(backend)

        droplet_id = 0
        state = core.particles
        state['cell origin'][0, droplet_id] = 1.1
        state['cell origin'][1, droplet_id] = 1.2

        # Act
        sut.boundary_condition(state['cell origin'])

        # Assert
        assert state['cell origin'][0, droplet_id] == 0
        assert state['cell origin'][1, droplet_id] == 0
