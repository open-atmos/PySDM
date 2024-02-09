"""
derived attributes providing amounts of light isotopes in water (1H and 16O)

        (water mass) = (
            moles_H2O * (2 * molar_mass_1H + molar_mass_16O) +
            moles_2H *  (molar_mass_1H + molar_mass_2H + molar_mass_16O) +
            moles_3H *  (molar_mass_1H + molar_mass_3H + molar_mass_16O) +
            moles_17O * (2 * molar_mass_1H + molar_mass_17O) +
            moles_18O * (2 * molar_mass_1H + molar_mass_18O)
        )

        moles_H2O = (
            water_mass
            - moles_2H *  (molar_mass_1H + molar_mass_2H + molar_mass_16O)
            - moles_3H *  (molar_mass_1H + molar_mass_3H + molar_mass_16O)
            - moles_17O * (2 * molar_mass_1H + molar_mass_17O)
            - moles_18O * (2 * molar_mass_1H + molar_mass_18O)
        ) / (2 * molar_mass_1H + molar_mass_16O)

        moles_1H = 2 * (moles_H2O + moles_17O + moles_18O) + moles_2H + moles_3H
        moles_16O =  .5 * (moles_2H + moles_3H) + moles_H2O
"""

from PySDM.attributes.impl import DerivedAttribute


class Helper(DerivedAttribute):
    def __init__(self, builder, name, attrs_to_multiplier):
        self.attrs_to_multiplier = attrs_to_multiplier
        super().__init__(
            builder=builder,
            name=name,
            dependencies=attrs_to_multiplier.keys(),
        )

    def recalculate(self):
        self.data.fill(0)
        for attr, mult in self.attrs_to_multiplier.items():
            self.data += (mult, "*", attr.data)


class MolesLightWater(Helper):
    def __init__(self, builder):
        const = builder.formulae.constants
        M_H2O = 2 * const.M_1H + const.M_16O
        super().__init__(
            builder=builder,
            name="moles light water",
            attrs_to_multiplier={
                builder.get_attribute("moles_2H"): -(
                    const.M_1H * const.M_2H + const.M_16O
                )
                / M_H2O,
                builder.get_attribute("moles_3H"): -(
                    const.M_1H * const.M_3H + const.M_16O
                )
                / M_H2O,
                builder.get_attribute("moles_17O"): -(2 * const.M_1H + const.M_17O)
                / M_H2O,
                builder.get_attribute("moles_18O"): -(2 * const.M_1H + const.M_18O)
                / M_H2O,
                builder.get_attribute("water mass"): 1 / M_H2O,
            },
        )


class Moles1H(Helper):
    def __init__(self, builder):
        super().__init__(
            builder=builder,
            name="moles_1H",
            attrs_to_multiplier={
                builder.get_attribute("moles_17O"): 2.0,
                builder.get_attribute("moles_18O"): 2.0,
                builder.get_attribute("moles_2H"): 1.0,
                builder.get_attribute("moles_3H"): 1.0,
                builder.get_attribute("moles light water"): 2.0,
            },
        )


class Moles16O(Helper):
    def __init__(self, builder):
        super().__init__(
            builder=builder,
            name="moles_16O",
            attrs_to_multiplier={
                builder.get_attribute("moles_2H"): 0.5,
                builder.get_attribute("moles_3H"): 0.5,
                builder.get_attribute("moles light water"): 1.0,
            },
        )
