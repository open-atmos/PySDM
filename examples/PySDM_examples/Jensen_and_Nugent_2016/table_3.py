# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import numpy as np

from PySDM.physics import si

NA = (
    np.asarray(
        [
            111800,
            68490,
            38400,
            21820,
            13300,
            8496,
            5486,
            3805,
            2593,
            1919,
            1278,
            998.4,
            777.9,
            519.5,
            400.5,
            376.9,
            265.3,
            212.4,
            137.8,
            121.4,
            100.9,
            122.2,
            50.64,
            38.3,
            55.47,
            21.45,
            12.95,
            43.23,
            26.26,
            30.5,
            4.385,
            4.372,
            4.465,
            4.395,
            4.427,
            4.411,
            0,
            0,
            0,
            4.522,
            0,
            4.542,
        ]
    )
    / si.m**3
)

RD = np.linspace(0.8, 9, num=len(NA), endpoint=True) * si.um
