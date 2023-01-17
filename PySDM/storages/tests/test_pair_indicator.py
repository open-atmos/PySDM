import numpy as np


def test_init(pair_indicator_class):
    # Arrange
    pair_indicator_instance = pair_indicator_class(6)

    # Assert
    assert len(pair_indicator_instance) == 6
    assert np.all(pair_indicator_instance.indicator.to_ndarray())
