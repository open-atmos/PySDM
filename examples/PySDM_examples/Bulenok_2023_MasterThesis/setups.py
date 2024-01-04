from PySDM_examples.Bulenok_2023_MasterThesis.utils import ProductsNames
from PySDM_examples.Srivastava_1982 import Settings
from PySDM_examples.Srivastava_1982.simulation import Simulation

from PySDM.dynamics import Collision
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import ConstantMass
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.collision_kernels import ConstantK
from PySDM.physics import si
from PySDM.products import SuperDropletCountPerGridbox, VolumeFirstMoment, ZerothMoment

dt = 1 * si.s
DV = 1 * si.m**3
drop_mass_0 = 1 * si.g

TOTAL_NUMBER = 1e12

NO_BOUNCE = ConstEb(1)


def make_settings(n_sd, total_number, dv, c, beta, frag_mass, backend_class):
    if total_number is None:
        total_number = TOTAL_NUMBER
    elif callable(total_number):
        total_number = total_number(n_sd)

    if dv is None:
        dv = DV
    elif callable(dv):
        dv = dv(n_sd)

    print()
    print("== Settings ==")
    print("n_sd", n_sd)
    print("total_number", total_number)
    print("dv", dv)
    print()

    return Settings(
        srivastava_c=c,
        srivastava_beta=beta,
        frag_mass=frag_mass,
        drop_mass_0=drop_mass_0,
        dt=dt,
        dv=dv,
        n_sds=(),
        total_number=total_number,
        backend_class=backend_class,
    )


def setup_simulation(settings, n_sd, seed, double_precision=True):
    products = (
        SuperDropletCountPerGridbox(name=ProductsNames.super_particle_count),
        VolumeFirstMoment(name=ProductsNames.total_volume),
        ZerothMoment(name=ProductsNames.total_number),
    )

    collision_rate = settings.srivastava_c + settings.srivastava_beta
    simulation = Simulation(
        n_steps=None,
        settings=settings,
        collision_dynamic=Collision(
            collision_kernel=ConstantK(a=collision_rate),
            coalescence_efficiency=ConstEc(settings.srivastava_c / collision_rate),
            breakup_efficiency=NO_BOUNCE,
            fragmentation_function=ConstantMass(c=settings.frag_mass / settings.rho),
            warn_overflows=False,
            adaptive=False,
        ),
        double_precision=double_precision,
    )
    particulator = simulation.build(n_sd, seed, products=products)

    return particulator


def setup_coalescence_only_sim(
    n_sd, backend_class, seed, double_precision=True, total_number=None, dv=None
):
    c = 0.5e-6 / si.s
    beta = 1e-15 / si.s
    frag_mass = -1 * si.g

    settings = make_settings(n_sd, total_number, dv, c, beta, frag_mass, backend_class)

    return setup_simulation(settings, n_sd, seed, double_precision)


def setup_breakup_only_sim(
    n_sd, backend_class, seed, double_precision=True, total_number=None, dv=None
):
    c = 1e-15 / si.s
    beta = 1e-9 / si.s
    frag_mass = 0.25 * si.g

    settings = make_settings(n_sd, total_number, dv, c, beta, frag_mass, backend_class)

    return setup_simulation(settings, n_sd, seed, double_precision)


def setup_coalescence_breakup_sim(
    n_sd, backend_class, seed, double_precision=True, total_number=None, dv=None
):
    c = 0.5e-6 / si.s
    beta = 1e-9 / si.s
    frag_mass = 0.25 * si.g

    settings = make_settings(n_sd, total_number, dv, c, beta, frag_mass, backend_class)

    return setup_simulation(settings, n_sd, seed, double_precision)
