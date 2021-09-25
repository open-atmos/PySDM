from functools import partial
from PySDM.attributes.impl.dummy_attribute import DummyAttributeImpl
from PySDM.attributes.physics.dry_volume import DryVolumeOrganic, DryVolume, DryVolumeDynamic, OrganicFraction
from PySDM.attributes.physics.hygroscopicity import Kappa, KappaTimesDryVolume
from PySDM.attributes.physics import (Multiplicities, Volume, Radius, DryRadius,
                                      TerminalVelocity, Temperature, Heat, CriticalVolume)
from PySDM.attributes.ice import FreezingTemperature, ImmersedSurfaceArea
from PySDM.attributes.numerics import CellID, CellOrigin, PositionInCell
from PySDM.attributes.chemistry import MoleAmount, Concentration, pH, HydrogenIonConcentration
from PySDM.physics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS
from PySDM.physics.surface_tension import Constant

attributes = {
    'n': lambda _: Multiplicities,
    'volume': lambda _: Volume,
    'dry volume organic': lambda dynamics: (
        DummyAttributeImpl('dry volume organic')
        if isinstance(dynamics['Condensation'].particulator.formulae.surface_tension, Constant)
        else DryVolumeOrganic
    ),
    'dry volume': lambda dynamics:
    DryVolumeDynamic if 'AqueousChemistry' in dynamics else DryVolume,
    'dry volume organic fraction': lambda dynamics: (
        DummyAttributeImpl('dry volume organic fraction')
        if isinstance(dynamics['Condensation'].particulator.formulae.surface_tension, Constant)
        else OrganicFraction
    ),
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
       for compound in AQUEOUS_COMPOUNDS},
    **{"conc_" + compound: partial(lambda _, c: Concentration(c), c=compound)
       for compound in AQUEOUS_COMPOUNDS},
    'pH': lambda _: pH,
    'conc_H': lambda _: HydrogenIonConcentration,
    'freezing temperature': lambda _: FreezingTemperature,
    'immersed surface area': lambda _: ImmersedSurfaceArea
}


def get_class(name, dynamics):
    return attributes[name](dynamics)
