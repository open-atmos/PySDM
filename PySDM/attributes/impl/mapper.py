"""
attribute name-class mapping logic (each new attribute must be added here)
"""
from functools import partial

from PySDM.attributes.chemistry import (
    Acidity,
    HydrogenIonConcentration,
    make_concentration_factory,
    make_mole_amount_factory,
)
from PySDM.attributes.ice import CoolingRate, FreezingTemperature, ImmersedSurfaceArea
from PySDM.attributes.impl.dummy_attribute import make_dummy_attribute_factory
from PySDM.attributes.numerics import CellID, CellOrigin, PositionInCell
from PySDM.attributes.physics import (
    Area,
    CriticalVolume,
    DryRadius,
    Heat,
    Multiplicities,
    Radius,
    Temperature,
    TerminalVelocity,
    Volume,
    WetToCriticalVolumeRatio,
)
from PySDM.attributes.physics.critical_supersaturation import CriticalSupersaturation
from PySDM.attributes.physics.dry_volume import (
    DryVolume,
    DryVolumeDynamic,
    DryVolumeOrganic,
    OrganicFraction,
)
from PySDM.attributes.physics.hygroscopicity import Kappa, KappaTimesDryVolume
from PySDM.dynamics.impl.chemistry_utils import AQUEOUS_COMPOUNDS
from PySDM.physics.surface_tension import Constant

attributes = {
    "n": lambda _, __: Multiplicities,
    "volume": lambda _, __: Volume,
    "dry volume organic": lambda dynamics, formulae: (
        make_dummy_attribute_factory("dry volume organic")
        if "Condensation" in dynamics
        and formulae.surface_tension.__name__ == Constant.__name__
        else DryVolumeOrganic
    ),
    "dry volume": lambda dynamics, formulae: DryVolumeDynamic
    if "AqueousChemistry" in dynamics
    else DryVolume,
    "dry volume organic fraction": lambda dynamics, formulae: (
        make_dummy_attribute_factory("dry volume organic fraction")
        if "Condensation" in dynamics
        and formulae.surface_tension.__name__ == Constant.__name__
        else OrganicFraction
    ),
    "kappa times dry volume": lambda _, __: KappaTimesDryVolume,
    "kappa": lambda _, __: Kappa,
    "radius": lambda _, __: Radius,
    "area": lambda _, __: Area,
    "dry radius": lambda _, __: DryRadius,
    "terminal velocity": lambda _, __: TerminalVelocity,
    "cell id": lambda _, __: CellID,
    "cell origin": lambda _, __: CellOrigin,
    "cooling rate": lambda _, __: CoolingRate,
    "position in cell": lambda _, __: PositionInCell,
    "temperature": lambda _, __: Temperature,
    "heat": lambda _, __: Heat,
    "critical volume": lambda _, __: CriticalVolume,
    **{
        "moles_"
        + compound: partial(lambda _, __, c: make_mole_amount_factory(c), c=compound)
        for compound in AQUEOUS_COMPOUNDS
    },
    **{
        "conc_"
        + compound: partial(lambda _, __, c: make_concentration_factory(c), c=compound)
        for compound in AQUEOUS_COMPOUNDS
    },
    "pH": lambda _, __: Acidity,
    "conc_H": lambda _, __: HydrogenIonConcentration,
    "freezing temperature": lambda _, __: FreezingTemperature,
    "immersed surface area": lambda _, __: ImmersedSurfaceArea,
    "critical supersaturation": lambda _, __: CriticalSupersaturation,
    "wet to critical volume ratio": lambda _, __: WetToCriticalVolumeRatio,
}


def get_class(name, dynamics, formulae):
    if name not in attributes:
        raise ValueError(
            f"Unknown attribute name: {name}; valid names: {', '.join(attributes)}"
        )
    return attributes[name](dynamics, formulae)
