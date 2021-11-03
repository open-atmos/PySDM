from functools import partial
from PySDM.attributes.impl.dummy_attribute import make_dummy_attribute_factory
from PySDM.attributes.physics.dry_volume import (DryVolumeOrganic, DryVolume, DryVolumeDynamic,
                                                 OrganicFraction)
from PySDM.attributes.physics.hygroscopicity import Kappa, KappaTimesDryVolume
from PySDM.attributes.physics import (Multiplicities, Volume, Radius, DryRadius,
                                      TerminalVelocity, Temperature, Heat, CriticalVolume)
from PySDM.attributes.ice import FreezingTemperature, ImmersedSurfaceArea
from PySDM.attributes.numerics import CellID, CellOrigin, PositionInCell
from PySDM.attributes.chemistry import (
    make_mole_amount_factory, make_concentration_factory, Acidity, HydrogenIonConcentration)
from PySDM.attributes.physics.critical_supersaturation import CriticalSupersaturation
from PySDM.physics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS
from PySDM.physics.surface_tension import Constant

attributes = {
    'n': lambda _: Multiplicities,
    'volume': lambda _: Volume,
    'dry volume organic': lambda dynamics: (
        make_dummy_attribute_factory('dry volume organic')
        if 'Condensation' in dynamics and isinstance(
            dynamics['Condensation'].particulator.formulae.surface_tension,
            Constant
        )
        else DryVolumeOrganic
    ),
    'dry volume': lambda dynamics:
    DryVolumeDynamic if 'AqueousChemistry' in dynamics else DryVolume,
    'dry volume organic fraction': lambda dynamics: (
        make_dummy_attribute_factory('dry volume organic fraction')
        if 'Condensation' in dynamics and isinstance(
            dynamics['Condensation'].particulator.formulae.surface_tension,
            Constant
        )
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
    **{"moles_" + compound: partial(lambda _, c: make_mole_amount_factory(c), c=compound)
       for compound in AQUEOUS_COMPOUNDS},
    **{"conc_" + compound: partial(lambda _, c: make_concentration_factory(c), c=compound)
       for compound in AQUEOUS_COMPOUNDS},
    'pH': lambda _: Acidity,
    'conc_H': lambda _: HydrogenIonConcentration,
    'freezing temperature': lambda _: FreezingTemperature,
    'immersed surface area': lambda _: ImmersedSurfaceArea,
    'critical supersaturation': lambda _: CriticalSupersaturation
}


def get_class(name, dynamics):
    if name not in attributes:
        raise ValueError(f"Unknown attribute name: {name}; valid names: {', '.join(attributes)}")
    return attributes[name](dynamics)
