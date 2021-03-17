"""
Created at 12.05.2020
"""

from PySDM.attributes.physics.multiplicities import Multiplicities
from PySDM.attributes.physics.volume import Volume
from PySDM.attributes.physics.dry_volume import DryVolume
from PySDM.attributes.physics.radius import Radius
from PySDM.attributes.physics.dry_radius import DryRadius
from PySDM.attributes.physics.terminal_velocity.terminal_velocity import TerminalVelocity
from PySDM.attributes.numerics.cell_id import CellID
from PySDM.attributes.numerics.cell_origin import CellOrigin
from PySDM.attributes.numerics.position_in_cell import PositionInCell
from PySDM.attributes.physics.temperature import Temperature
from PySDM.attributes.physics.heat import Heat
from PySDM.attributes.physics.critical_radius import CriticalVolume
from PySDM.attributes.chemistry.mole_amount import MoleAmount
from PySDM.attributes.chemistry.concentration import Concentration
from PySDM.attributes.chemistry.pH import pH
from PySDM.dynamics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS

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
    'heat': Heat,
    'critical volume': CriticalVolume,
    **{"moles_" + compound: MoleAmount(compound) for compound in AQUEOUS_COMPOUNDS.keys()},
    **{"conc_" + compound: Concentration(compound) for compound in AQUEOUS_COMPOUNDS.keys()},
    'pH': pH
}


def get_class(name):
    return attributes[name]
