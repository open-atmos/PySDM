"""
Created at 12.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .droplet.multiplicities import Multiplicities
from .droplet.volume import Volume
from .droplet.dry_volume import DryVolume
from .droplet.radius import Radius
from .droplet.dry_radius import DryRadius
from .droplet.terminal_velocity import TerminalVelocity
from .cell.cell_id import CellID

# TODO doubled information
attributes = {
    'n': Multiplicities,
    'volume': Volume,
    'dry volume': DryVolume,
    'radius': Radius,
    'dry radius': DryRadius,
    'terminal velocity': TerminalVelocity,
    'cell id': CellID
}


def get_class(name):
    return attributes[name]
