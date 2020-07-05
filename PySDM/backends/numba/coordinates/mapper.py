"""
Created at 06.05.2020
"""

from . import volume as coord_volume
from . import volume_logarithm as coord_volume_logarithm


def get(coord):
    if coord == 'volume':
        coord = coord_volume
    elif coord == 'volume logarithm':
        coord = coord_volume_logarithm
    else:
        raise ValueError(f"Unknown {coord} coordinates. Please chose one from: ['volume', 'volume logarithm']")
    x = coord.x
    volume = coord.volume
    dx_dt = coord.dx_dt

    return dx_dt, volume, x
