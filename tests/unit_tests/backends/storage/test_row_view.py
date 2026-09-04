"""
Test that row_view is created properly and that changes made in row_view affect the parent storage.
"""

import numpy as np


def test_row_view(backend_instance_with_jax):
    backend = backend_instance_with_jax
    output = backend.Storage.from_ndarray(np.zeros((3, 16)))
    row = output.row_view(1)
    row.fill(5)

    expected = [np.zeros(16), np.full(16, 5), np.zeros(16)]

    # Assert
    np.testing.assert_array_equal(output.to_ndarray(), expected)
