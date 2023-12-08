import PySDM.products.size_spectral.arbitrary_moment as am
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Collision
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity


def make_core(settings, coal_eff):
    backend = CPU

    env = Box(dv=settings.dv, dt=settings.dt)
    builder = Builder(
        n_sd=settings.n_sd, backend=backend(settings.formulae), environment=env
    )
    env["rhod"] = 1.0
    attributes = {}
    attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
        settings.spectrum
    ).sample(settings.n_sd)
    collision = Collision(
        collision_kernel=settings.kernel,
        coalescence_efficiency=coal_eff,
        breakup_efficiency=settings.break_eff,
        fragmentation_function=settings.fragmentation,
        adaptive=settings.adaptive,
    )
    builder.add_dynamic(collision)
    M0 = am.make_arbitrary_moment_product(rank=0, attr="volume", attr_unit="m^3")
    M1 = am.make_arbitrary_moment_product(rank=1, attr="volume", attr_unit="m^3")
    M2 = am.make_arbitrary_moment_product(rank=2, attr="volume", attr_unit="m^3")
    products = (M0(name="M0"), M1(name="M1"), M2(name="M2"))
    return builder.build(attributes, products)
