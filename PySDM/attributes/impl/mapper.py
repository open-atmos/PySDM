"""
attribute name-class mapping logic (each new attribute must be added here)
"""

from functools import partial

from PySDM.attributes.chemistry import (
    Acidity,
    HydrogenIonConcentration,
    make_concentration_factory,
)
from PySDM.attributes.ice import CoolingRate, FreezingTemperature, ImmersedSurfaceArea
from PySDM.attributes.impl.dummy_attribute import make_dummy_attribute_factory
from PySDM.attributes.impl.mole_amount import make_mole_amount_factory
from PySDM.attributes.isotopes import Moles1H, Moles16O, MolesLightWater
from PySDM.attributes.isotopes.delta import make_delta_factory
from PySDM.attributes.numerics import CellID, CellOrigin, PositionInCell
from PySDM.attributes.physics import (
    Area,
    CriticalVolume,
    DryRadius,
    Heat,
    Multiplicities,
    Radius,
    RelativeFallVelocity,
    SquareRootOfRadius,
    Temperature,
    TerminalVelocity,
    Volume,
    WaterMass,
    WetToCriticalVolumeRatio,
    ReynoldsNumber,
)
from PySDM.attributes.physics.critical_supersaturation import CriticalSupersaturation
from PySDM.attributes.physics.dry_volume import (
    DryVolume,
    DryVolumeDynamic,
    DryVolumeOrganic,
    OrganicFraction,
)
from PySDM.attributes.physics.equilibrium_supersaturation import (
    EquilibriumSupersaturation,
)
from PySDM.attributes.physics.hygroscopicity import Kappa, KappaTimesDryVolume
from PySDM.attributes.physics.relative_fall_velocity import RelativeFallMomentum
from PySDM.dynamics.impl.chemistry_utils import AQUEOUS_COMPOUNDS
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES
from PySDM.physics.surface_tension import Constant
from PySDM.physics.ventilation import Neglect

attributes = {
    "multiplicity": lambda _, __: Multiplicities,
    "volume": lambda _, __: Volume,
    "dry volume organic": lambda dynamics, formulae: (
        make_dummy_attribute_factory("dry volume organic")
        if formulae.surface_tension.__name__ == Constant.__name__
        else DryVolumeOrganic
    ),
    "dry volume": lambda dynamics, formulae: (
        DryVolumeDynamic if "AqueousChemistry" in dynamics else DryVolume
    ),
    "dry volume organic fraction": lambda dynamics, formulae: (
        make_dummy_attribute_factory("dry volume organic fraction")
        if formulae.surface_tension.__name__ == Constant.__name__
        else OrganicFraction
    ),
    "kappa times dry volume": lambda _, __: KappaTimesDryVolume,
    "kappa": lambda _, __: Kappa,
    "radius": lambda _, __: Radius,
    "square root of radius": lambda _, __: SquareRootOfRadius,
    "area": lambda _, __: Area,
    "dry radius": lambda _, __: DryRadius,
    "terminal velocity": lambda _, __: TerminalVelocity,
    "relative fall momentum": lambda dynamics, __: (
        RelativeFallMomentum
        if "RelaxedVelocity" in dynamics
        else make_dummy_attribute_factory("relative fall momentum", warn=True)
        # note: could eventually make an attribute that calculates momentum
        # from terminal velocity instead when no RelaxedVelocity dynamic is present
    ),
    "relative fall velocity": lambda dynamics, __: (
        RelativeFallVelocity if "RelaxedVelocity" in dynamics else TerminalVelocity
    ),
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
    "equilibrium supersaturation": lambda _, __: EquilibriumSupersaturation,
    "wet to critical volume ratio": lambda _, __: WetToCriticalVolumeRatio,
    "water mass": lambda _, __: WaterMass,
    **{
        "moles_"
        + isotope: partial(lambda _, __, c: make_mole_amount_factory(c), c=isotope)
        for isotope in HEAVY_ISOTOPES
    },
    **{
        "delta_" + isotope: partial(lambda _, __, c: make_delta_factory(c), c=isotope)
        for isotope in HEAVY_ISOTOPES
    },
    "moles_1H": lambda _, __: Moles1H,
    "moles_16O": lambda _, __: Moles16O,
    "moles light water": lambda _, __: MolesLightWater,
    "Reynolds number": lambda _, formulae: (
        make_dummy_attribute_factory("Reynolds number")
        if formulae.ventilation.__name__ == Neglect.__name__
        else ReynoldsNumber
    ),
}


def get_class(name, dynamics, formulae):
    if name not in attributes:
        raise ValueError(
            f"Unknown attribute name: {name}; valid names: {', '.join(sorted(attributes))}"
        )
    return attributes[name](dynamics, formulae)
