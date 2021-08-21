from PySDM.attributes.physics import (Multiplicities, Volume, DryVolumeDynamic, DryVolumeStatic, Radius, DryRadius,
                                      TerminalVelocity, Temperature, Heat, CriticalVolume)
from PySDM.attributes.ice import FreezingTemperature, SpheroidMass
from PySDM.attributes.numerics import CellID, CellOrigin, PositionInCell
from PySDM.attributes.chemistry import MoleAmount, Concentration, pH, HydrogenIonConcentration
from PySDM.physics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS
from functools import partial

attributes = {
    'n': lambda _: Multiplicities,
    'volume': lambda _: Volume,
    'dry volume': lambda dynamics: DryVolumeDynamic if 'AqueousChemistry' in dynamics else DryVolumeStatic,
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
