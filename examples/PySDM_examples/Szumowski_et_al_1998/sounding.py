import numpy as np

from PySDM.physics import constants as const
from PySDM.physics import si

pressure = (
    np.array(
        (
            1014,
            1010,
            1000,
            990,
            975,
            960,
            950,
            925,
            900,
            875,
            850,
            825,
            800,
            790,
            775,
            765,
            755,
            745,
            730,
            715,
            700,
        )
    )
    * si.hPa
)

temperature = (
    np.array(
        (
            25.2,
            24.8,
            23.6,
            22.5,
            21.8,
            20.5,
            19.9,
            18.2,
            16.8,
            14.8,
            13.3,
            11.9,
            11.0,
            11.3,
            10.9,
            11.2,
            10.2,
            11.0,
            11.2,
            10.0,
            8.8,
        )
    )
    * si.kelvin
    + const.T0
)

mixing_ratio = (
    np.array(
        (
            14.5,
            14.5,
            14.5,
            14.0,
            13.7,
            13.9,
            13.9,
            10.3,
            10.3,
            10.0,
            9.9,
            8.9,
            7.9,
            4.0,
            2.3,
            1.2,
            1.2,
            0.9,
            0.6,
            2.0,
            1.6,
        )
    )
    * si.g
    / si.kg
)
