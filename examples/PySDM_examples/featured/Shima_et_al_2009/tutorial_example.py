from PySDM.backends import CPU
from PySDM import Particulator
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products import ParticleVolumeVersusRadiusLogarithmSpectrum, WallTime


def run(settings, observers=()):
    environment = Box(
        dv=settings.dv,
        dt=settings.dt,
        backend=CPU(formulae=settings.formulae),
    )
    attributes = {}
    sampling = ConstantMultiplicity(settings.spectrum)
    attributes["volume"], attributes["multiplicity"] = sampling.sample_deterministic(
        settings.n_sd
    )
    products = (
        ParticleVolumeVersusRadiusLogarithmSpectrum(
            settings.radius_bins_edges, name="dv/dlnr"
        ),
        WallTime(),
    )
    particulator = Particulator(
        n_sd=settings.n_sd,
        environment=environment,
        dynamics=(
            Coalescence(collision_kernel=settings.kernel, adaptive=settings.adaptive),
        ),
        attributes=attributes,
        products=products,
    )
    if hasattr(settings, "u_term") and "terminal velocity" in particulator.attributes:
        particulator.attributes["terminal velocity"].approximation = settings.u_term(
            particulator
        )

    for observer in observers:
        particulator.observers.append(observer)

    vals = {}
    particulator.products["wall time"].reset()
    for step in settings.output_steps:
        particulator.run(step - particulator.n_steps)
        vals[step] = particulator.products["dv/dlnr"].get()[0]
        vals[step][:] *= settings.rho

    exec_time = particulator.products["wall time"].get()
    return vals, exec_time
