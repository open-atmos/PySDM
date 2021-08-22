from PySDM.attributes.physics import (Multiplicities, Volume, DryVolumeDynamic, DryVolumeStatic, Radius, DryRadius,
                                      TerminalVelocity, Temperature, Heat, CriticalVolume)
from PySDM.attributes.ice import FreezingTemperature, SpheroidMass
from PySDM.attributes.numerics import CellID, CellOrigin, PositionInCell
from PySDM.attributes.chemistry import MoleAmount, Concentration, pH, HydrogenIonConcentration
from PySDM.attributes.physics.multiplicities import Multiplicities
from PySDM.attributes.physics.volume import Volume
from PySDM.attributes.physics.dry_volume import (DryVolumeOrganic, DryVolumeInorganic,
                                                 DryVolume, DryVolumeInorganicDynamic,
                                                 OrganicFraction)
from PySDM.attributes.physics.hygroscopicity import Kappa, KappaTimesDryVolume
from PySDM.attributes.physics.radius import Radius
from PySDM.attributes.physics.dry_radius import DryRadius
from PySDM.attributes.physics.terminal_velocity import TerminalVelocity
from PySDM.attributes.numerics.cell_id import CellID
from PySDM.attributes.numerics.cell_origin import CellOrigin
from PySDM.attributes.numerics.position_in_cell import PositionInCell
from PySDM.attributes.physics.temperature import Temperature
from PySDM.attributes.physics.heat import Heat
from PySDM.attributes.physics.critical_volume import CriticalVolume
from PySDM.attributes.chemistry.mole_amount import MoleAmount
from PySDM.attributes.chemistry.concentration import Concentration
from PySDM.attributes.chemistry.pH import pH
from PySDM.attributes.chemistry.hydrogen_ion_concentration import HydrogenIonConcentration
from PySDM.physics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS
from functools import partial

attributes = {
    'n': lambda _: Multiplicities,
    'volume': lambda _: Volume,
    'dry volume organic': lambda _: DryVolumeOrganic,
    'dry volume inorganic': lambda dynamics:
        DryVolumeInorganicDynamic if 'AqueousChemistry' in dynamics else DryVolumeInorganic,
    'dry volume': lambda _: DryVolume,
    'dry volume organic fraction': lambda _: OrganicFraction,
    'kappa times dry volume': lambda _: KappaTimesDryVolume,
    'kappa': lambda _: Kappa,
    'radius': lambda _: Radius,
    'dry radius': lambda _: DryRadius,
    'terminal velocity': lambda _: TerminalVelocity,
    'cell id': lambda _: CellID,
    'cell origin': lambda _: CellOrigin,
    'position in cell': lambda _: PositionInCell,
    'temperature': lambda _: Temperature,
    'heat': lambda _: Heat,
    'critical volume': lambda _: CriticalVolume,
    **{"moles_" + compound: partial(lambda _, c: MoleAmount(c), c=compound)
       for compound in AQUEOUS_COMPOUNDS.keys()},
    **{"conc_" + compound: partial(lambda _, c: Concentration(c), c=compound)
       for compound in AQUEOUS_COMPOUNDS.keys()},
    'pH': lambda _: pH,
    'conc_H': lambda _: HydrogenIonConcentration,
    'freezing temperature': lambda _: FreezingTemperature,
    'spheroid mass': lambda _: SpheroidMass
}


def get_class(name, dynamics):
    return attributes[name](dynamics)
