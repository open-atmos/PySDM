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
    common_args = {
        "attr": "volume",
        "attr_unit": "m^3",
        "skip_division_by_m0": True,
        "skip_division_by_dv": True,
    }
    products = tuple(
        am.make_arbitrary_moment_product(rank=rank, **common_args)(name=f"M{rank}")
        for rank in range(3)
    )
    return builder.build(attributes, products)
