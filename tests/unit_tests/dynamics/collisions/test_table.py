"""
Test the table function in the Hall class.
"""

from PySDM.dynamics.collisions.collision_kernels.hall import Hall


def test_table():
    hall = Hall()
    collector_radius_m = 2e-5
    collected_radius_m = 1e-6
    result1 = hall.table(collected_radius_m, collector_radius_m)
    result2 = hall.table(collector_radius_m, collected_radius_m)
    assert result1 == result2 == 1e-4, "Test failed"
