"""expansion-chamber environment tests"""

import numpy as np

from PySDM import Builder
from PySDM.environments import ExpansionChamber


def test_expansion_chamber(backend_instance):
    """a minimal instantiation test"""
    # arrange
    env = ExpansionChamber(
        dt=np.nan,
        volume=np.nan,
        initial_pressure=np.nan,
        initial_temperature=np.nan,
        initial_relative_humidity=np.nan,
        delta_pressure=np.nan,
        delta_time=np.nan,
    )
    particulator = Builder(n_sd=1, backend=backend_instance, environment=env).build(
        attributes={
            "multiplicity": np.asarray([1]),
            "water mass": np.asarray([np.nan]),
        },
        products=(),
    )

    # act

    # assert
    # TODO #1492
