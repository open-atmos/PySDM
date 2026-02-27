from ..impl import (
    register_attribute,
    ExtensiveAttribute,
    DummyAttribute,
    DerivedAttribute,
)


class DropwiseTendency(DummyAttribute):
    def __init__(self, builder, *, name):
        super().__init__(builder, name=name + " tendency")
        builder.particulator.initialisers.append(self)

    def setup(self):
        self.data.data[:] = 0.0


class DropwiseThermodynamicVarRelaxed(
    DummyAttribute
):  # TODO: it should not be extensive - what happens with Qk and Tk when two droplets collide!
    def __init__(self, builder, *, name):
        super().__init__(builder, name=name)
        builder.particulator.initialisers.append(self)

    def setup(self):
        self.data.data[:] = self.particulator.environment[
            {
                "dropwise water vapour mixing ratio": "water_vapour_mixing_ratio",
                "dropwise dry air potential temperature": "thd",
            }[self.name]
        ][self.particulator.attributes["cell id"].data]


class DropwiseThermodynamicVarAmbient(DropwiseThermodynamicVarRelaxed):
    def __init__(self, builder, *, name):
        super().__init__(builder, name=name)
        builder.particulator.observers.append(
            self
        )  # TODO: check if this is not too early/late for the update?

    def notify(self):
        self.setup()


def make_tendency_factory(name):
    def _factory(builder):
        return DropwiseTendency(builder, name=name)

    return _factory


def make_relaxed_var_factory(name):
    def _factory(builder):
        return DropwiseThermodynamicVarRelaxed(builder, name=name)

    return _factory


def make_ambient_var_factory(name):
    def _factory(builder):
        return DropwiseThermodynamicVarAmbient(builder, name=name)

    return _factory


for var in (
    "dropwise water vapour mixing ratio",
    "dropwise dry air potential temperature",
):
    register_attribute(name=f"{var} tendency")(make_tendency_factory(var))
    register_attribute(
        name=var, variant=lambda dynamics, _: "DropLocalThermodynamics" in dynamics
    )(make_relaxed_var_factory(var))
    register_attribute(
        name=var, variant=lambda dynamics, _: "DropLocalThermodynamics" not in dynamics
    )(make_ambient_var_factory(var))


# TODO: shouldn't it be defined as mass rather than mixing ratio? (like for multiplicity)
