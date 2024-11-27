from collections import namedtuple

import numpy as np
from PySDM import Builder
from PySDM import products as PySDM_products
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si


def run_expansion(
    formulae,
    aerosol,
    n_sd_per_mode,
    RH0=0.7,
    dp=500 * si.hPa,
    T0=296 * si.K,
    p0=1000 * si.hPa,
    dz=10 * si.m,
    t_lift=2 * si.s,
    t_max=5 * si.s,
):

    # calculate w, dt, n_steps
    H = 8500 * si.m  # atmospheric scale height
    z_lift = -H * np.log(dp / p0)
    w_lift = z_lift / t_lift
    w = lambda t: w_lift if t < t_lift else 1e-5
    dt = dz / w_lift
    n_steps = int(np.ceil(t_max / dt))
    # print(f"z_lift={z_lift}")
    # print(f"w_lift={w_lift}")
    # print(f"dz={dz}")
    # print(f"dt={dt}")
    print(f"n_steps={n_steps}")

    dry_radius_bin_edges = np.geomspace(50 * si.nm, 2000 * si.nm, 40, endpoint=False)
    wet_radius_bin_edges = np.geomspace(1 * si.um, 40 * si.um, 40, endpoint=False)
    products = (
        PySDM_products.WaterMixingRatio(unit="g/kg", name="liquid_water_mixing_ratio"),
        PySDM_products.PeakSupersaturation(name="smax"),
        PySDM_products.AmbientRelativeHumidity(name="RH"),
        PySDM_products.AmbientTemperature(name="T"),
        PySDM_products.AmbientPressure(name="p"),
        PySDM_products.AmbientWaterVapourMixingRatio(
            unit="g/kg", name="water_vapour_mixing_ratio"
        ),
        PySDM_products.ParcelDisplacement(name="z"),
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
    )

    const = formulae.constants
    pv0 = RH0 * formulae.saturation_vapour_pressure.pvs_water(T0)

    env = Parcel(
        dt=dt,
        mass_of_dry_air=1e3 * si.kg,
        p0=p0,
        initial_water_vapour_mixing_ratio=const.eps * pv0 / (p0 - pv0),
        w=w,
        T0=T0,
    )

    n_sd = n_sd_per_mode * len(aerosol.modes)

    builder = Builder(backend=CPU(formulae), n_sd=n_sd, environment=env)
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
        specific_concentration = concentration / builder.formulae.constants.rho_STP
        attributes["multiplicity"] = np.append(
            attributes["multiplicity"],
            specific_concentration * builder.particulator.environment.mass_of_dry_air,
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
