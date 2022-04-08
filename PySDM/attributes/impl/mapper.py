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
    "n": lambda _: Multiplicities,
    "volume": lambda _: Volume,
    "dry volume organic": lambda dynamics: (
        make_dummy_attribute_factory("dry volume organic")
        if "Condensation" in dynamics
        and (
            dynamics["Condensation"].particulator.formulae.surface_tension.__name__
            == Constant.__name__
        )
        else DryVolumeOrganic
    ),
    "dry volume": lambda dynamics: DryVolumeDynamic
    if "AqueousChemistry" in dynamics
    else DryVolume,
    "dry volume organic fraction": lambda dynamics: (
        make_dummy_attribute_factory("dry volume organic fraction")
        if "Condensation" in dynamics
        and (
            dynamics["Condensation"].particulator.formulae.surface_tension.__name__
            == Constant.__name__
        )
        else OrganicFraction
    ),
    "kappa times dry volume": lambda _: KappaTimesDryVolume,
    "kappa": lambda _: Kappa,
    "radius": lambda _: Radius,
    "area": lambda _: Area,
    "dry radius": lambda _: DryRadius,
    "terminal velocity": lambda _: TerminalVelocity,
    "cell id": lambda _: CellID,
    "cell origin": lambda _: CellOrigin,
    "cooling rate": lambda _: CoolingRate,
    "position in cell": lambda _: PositionInCell,
    "temperature": lambda _: Temperature,
    "heat": lambda _: Heat,
    "critical volume": lambda _: CriticalVolume,
    **{
        "moles_"
        + compound: partial(lambda _, c: make_mole_amount_factory(c), c=compound)
        for compound in AQUEOUS_COMPOUNDS
    },
    **{
        "conc_"
        + compound: partial(lambda _, c: make_concentration_factory(c), c=compound)
        for compound in AQUEOUS_COMPOUNDS
    },
    "pH": lambda _: Acidity,
    "conc_H": lambda _: HydrogenIonConcentration,
    "freezing temperature": lambda _: FreezingTemperature,
    "immersed surface area": lambda _: ImmersedSurfaceArea,
    "critical supersaturation": lambda _: CriticalSupersaturation,
}


def get_class(name, dynamics):
    if name not in attributes:
        raise ValueError(
            f"Unknown attribute name: {name}; valid names: {', '.join(attributes)}"
        )
    return attributes[name](dynamics)
