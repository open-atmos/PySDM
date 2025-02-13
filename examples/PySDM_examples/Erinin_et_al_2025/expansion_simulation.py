from collections import namedtuple

import numpy as np
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.dynamics import (
    AmbientThermodynamics,
    Condensation,
    HomogeneousLiquidNucleation,
)
from PySDM.environments import ExpansionChamber
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si


def run_expansion(
    *,
    formulae,
    aerosol,
    n_sd_per_mode,
    n_sd_homo_liq_nucleation=100,
    RH0=0.7,
    T0=296 * si.K,
    p0=1000 * si.hPa,
    pf=500 * si.hPa,
    delta_time=2 * si.s,
    total_time=4 * si.s,
    dt=0.1 * si.s,
    volume=0.14 * si.m**3,
    products=None,
):
    n_steps = int(np.ceil(total_time / dt))

    env = ExpansionChamber(
        dt=dt,
        initial_pressure=p0,
        delta_pressure=pf - p0,
        initial_temperature=T0,
        initial_relative_humidity=RH0,
        delta_time=delta_time,
        volume=volume,
    )

    n_sd = n_sd_per_mode * len(aerosol.modes) + n_sd_homo_liq_nucleation

    builder = Builder(
        backend=CPU(formulae, override_jit_flags={"parallel": False}),
        n_sd=n_sd,
        environment=env,
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation(adaptive=True))
    builder.add_dynamic(HomogeneousLiquidNucleation())
    builder.request_attribute("critical supersaturation")

    attributes = {
        k: np.empty(0) for k in ("dry volume", "kappa times dry volume", "multiplicity")
    }
    for mode in aerosol.modes:
        kappa, spectrum = mode["kappa"]["Constant"], mode["spectrum"]
        r_dry, concentration = ConstantMultiplicity(spectrum).sample(n_sd_per_mode)
        v_dry = builder.formulae.trivia.volume(radius=r_dry)
        attributes["multiplicity"] = np.append(
            attributes["multiplicity"],
            concentration * builder.particulator.environment.dv,
        )
        attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
        attributes["kappa times dry volume"] = np.append(
            attributes["kappa times dry volume"], v_dry * kappa
        )

    r_wet = equilibrate_wet_radii(
        r_dry=builder.formulae.trivia.radius(volume=attributes["dry volume"]),
        environment=builder.particulator.environment,
        kappa_times_dry_volume=attributes["kappa times dry volume"],
    )
    attributes["volume"] = builder.formulae.trivia.volume(radius=r_wet)

    particulator = builder.build(
        attributes={
            k: np.pad(
                array=v,
                pad_width=(0, n_sd_homo_liq_nucleation),
                mode="constant",
                constant_values=np.nan if k == "multiplicity" else 0,
            )
            for k, v in attributes.items()
        },
        products=products or (),
    )

    output = {product.name: [] for product in particulator.products.values()}
    output_attributes = {
        k: []
        for k in (
            "multiplicity",
            "volume",
            "critical volume",
            "critical supersaturation",
        )
    }

    for _ in range(n_steps):
        particulator.run(steps=1)
        for product in particulator.products.values():
            if product.name in ("dry:dN/dR", "wet:dN/dR"):
                continue
            value = product.get()
            if product.name == "t":
                output[product.name].append(value)
            else:
                output[product.name].append(value[0])
        mult = particulator.attributes["multiplicity"].to_ndarray(raw=True)
        for key, attr in output_attributes.items():
            if key == "multiplicity":
                attr.append(mult)
                continue
            data = particulator.attributes[key].to_ndarray(raw=True)
            data[mult == 0] = np.nan
            attr.append(data)

    dry_spectrum = particulator.products["dry:dN/dR"].get()
    wet_spectrum = particulator.products["wet:dN/dR"].get()

    Output = namedtuple(
        "Output",
        (
            "profile",
            "attributes",
            "aerosol",
            "dry_spectrum",
            "wet_spectrum",
            "units",
        ),
    )
    return Output(
        profile=output,
        attributes=output_attributes,
        aerosol=aerosol,
        dry_spectrum=dry_spectrum,
        wet_spectrum=wet_spectrum,
        units={
            key: (
                product.unit[2:]
                if product.unit[0:2] == "1 "
                else product.unit[4:] if product.unit[0:4] == "1.0 " else product.unit
            ).replace("**", "^")
            for key, product in particulator.products.items()
        },
    )
