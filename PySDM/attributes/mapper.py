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
from .droplet.terminal_velocity.terminal_velocity import TerminalVelocity
from .cell.cell_id import CellID
from .cell.cell_origin import CellOrigin
from .cell.position_in_cell import PositionInCell
from .droplet.temperature import Temperature
from .droplet.critical_radius import CriticalRadius

# TODO doubled information
attributes = {
    'n': Multiplicities,
    'volume': Volume,
    'dry volume': DryVolume,
    'radius': Radius,
    'dry radius': DryRadius,
    'terminal velocity': TerminalVelocity,
    'cell id': CellID,
    'cell origin': CellOrigin,
    'position in cell': PositionInCell,
    'temperature': Temperature,
    'critical radius': CriticalRadius}


def get_class(name):
    return attributes[name]
