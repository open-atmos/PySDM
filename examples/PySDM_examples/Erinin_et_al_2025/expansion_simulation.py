from collections import namedtuple

import numpy as np
from PySDM import Builder
from PySDM import products as PySDM_products
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import ExpansionChamber
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si


def run_expansion(
    formulae,
    aerosol,
    n_sd_per_mode,
    RH0=0.7,
    T0=296 * si.K,
    Tf=261 * si.K,
    p0=1000 * si.hPa,
    pf=500 * si.hPa,
    t_lift=2 * si.s,
    t_max=4 * si.s,
    dt=0.1 * si.s,
    dv=0.14 * si.m**3,
):
    n_steps = int(np.ceil(t_max / dt))
    dry_radius_bin_edges = np.geomspace(50 * si.nm, 2000 * si.nm, 40, endpoint=False)
    wet_radius_bin_edges = np.geomspace(1 * si.um, 40 * si.um, 40, endpoint=False)
    products = (
        PySDM_products.WaterMixingRatio(unit="g/kg", name="liquid_water_mixing_ratio"),
        PySDM_products.PeakSupersaturation(name="s"),
        PySDM_products.AmbientRelativeHumidity(name="RH"),
        PySDM_products.AmbientTemperature(name="T"),
        PySDM_products.AmbientPressure(name="p"),
        PySDM_products.AmbientWaterVapourMixingRatio(
            unit="g/kg", name="water_vapour_mixing_ratio"
        ),
        PySDM_products.Time(name="t"),
        PySDM_products.ParticleSizeSpectrumPerVolume(
            name="dry:dN/dR",
            unit="m^-3 m^-1",
            radius_bins_edges=dry_radius_bin_edges,
            dry=True,
        ),
        PySDM_products.ParticleSizeSpectrumPerVolume(
            name="wet:dN/dR",
            unit="m^-3 m^-1",
            radius_bins_edges=wet_radius_bin_edges,
            dry=False,
        ),
        PySDM_products.ActivatedEffectiveRadius(
            name="reff", unit="um", count_activated=True, count_unactivated=False
        ),
    )

    env = ExpansionChamber(
        dt=dt,
        initial_pressure=p0,
        delta_pressure=pf - p0,
        initial_temperature=T0,
        delta_temperature=Tf - T0,
        initial_relative_humidity=RH0,
        delta_time=t_lift,
        volume=dv,
    )

    n_sd = n_sd_per_mode * len(aerosol.modes)

    builder = Builder(
        backend=CPU(formulae, override_jit_flags={"parallel": False}),
        n_sd=n_sd,
        environment=env,
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(Condensation())
    builder.request_attribute("critical supersaturation")

    attributes = {
        k: np.empty(0) for k in ("dry volume", "kappa times dry volume", "multiplicity")
    }
    for i, mode in enumerate(aerosol.modes):
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

    particulator = builder.build(attributes, products=products)

    output = {product.name: [] for product in particulator.products.values()}
    output_attributes = {
        "multiplicity": tuple([] for _ in range(particulator.n_sd)),
        "volume": tuple([] for _ in range(particulator.n_sd)),
        "critical volume": tuple([] for _ in range(particulator.n_sd)),
        "critical supersaturation": tuple([] for _ in range(particulator.n_sd)),
    }

    for _ in range(n_steps):
        particulator.run(steps=1)
        for product in particulator.products.values():
            if product.name == "dry:dN/dR" or product.name == "wet:dN/dR":
                continue
            value = product.get()
            if product.name == "t":
                output[product.name].append(value)
            else:
                output[product.name].append(value[0])
        for key, attr in output_attributes.items():
            attr_data = particulator.attributes[key].to_ndarray()
            for drop_id in range(particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])

    dry_spectrum = particulator.products["dry:dN/dR"].get()
    wet_spectrum = particulator.products["wet:dN/dR"].get()

    Output = namedtuple(
        "Output",
        [
            "profile",
            "attributes",
            "aerosol",
            "dry_radius_bin_edges",
            "dry_spectrum",
            "wet_radius_bin_edges",
            "wet_spectrum",
        ],
    )
    return Output(
        profile=output,
        attributes=output_attributes,
        aerosol=aerosol,
        dry_radius_bin_edges=dry_radius_bin_edges,
        dry_spectrum=dry_spectrum,
        wet_radius_bin_edges=wet_radius_bin_edges,
        wet_spectrum=wet_spectrum,
    )
