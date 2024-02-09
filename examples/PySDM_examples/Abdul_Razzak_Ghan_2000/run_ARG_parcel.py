from collections import namedtuple

import numpy as np
from PySDM_examples.Abdul_Razzak_Ghan_2000.aerosol import CONSTANTS_ARG, AerosolARG

from PySDM import Builder, Formulae
from PySDM import products as PySDM_products
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si


def run_parcel(
    w,
    sol2,
    N2,
    rad2,
    n_sd_per_mode,
    RH0=1.0,
    T0=294 * si.K,
    p0=1e5 * si.Pa,
    n_steps=50,
    mass_of_dry_air=1e3 * si.kg,
    dt=2 * si.s,
):
    products = (
        PySDM_products.WaterMixingRatio(unit="g/kg", name="liquid water mixing ratio"),
        PySDM_products.PeakSupersaturation(name="S max"),
        PySDM_products.AmbientRelativeHumidity(name="RH"),
        PySDM_products.ParcelDisplacement(name="z"),
    )

    formulae = Formulae(constants=CONSTANTS_ARG)
    const = formulae.constants
    pv0 = RH0 * formulae.saturation_vapour_pressure.pvs_Celsius(T0 - const.T0)

    env = Parcel(
        dt=dt,
        mass_of_dry_air=mass_of_dry_air,
        p0=p0,
        initial_water_vapour_mixing_ratio=const.eps * pv0 / (p0 - pv0),
        w=w,
        T0=T0,
    )

    aerosol = AerosolARG(
        M2_sol=sol2, M2_N=N2, M2_rad=rad2, water_molar_volume=const.Mv / const.rho_w
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
        kappa, spectrum = mode["kappa"]["CompressedFilmOvadnevaite"], mode["spectrum"]
        r_dry, concentration = ConstantMultiplicity(spectrum).sample(n_sd_per_mode)
        v_dry = builder.formulae.trivia.volume(radius=r_dry)
        specific_concentration = concentration / builder.formulae.constants.rho_STP
        attributes["multiplicity"] = np.append(
            attributes["multiplicity"], specific_concentration * env.mass_of_dry_air
        )
        attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
        attributes["kappa times dry volume"] = np.append(
            attributes["kappa times dry volume"], v_dry * kappa
        )

    r_wet = equilibrate_wet_radii(
        r_dry=builder.formulae.trivia.radius(volume=attributes["dry volume"]),
        environment=env,
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
            value = product.get()
            output[product.name].append(value[0])
        for key, attr in output_attributes.items():
            attr_data = particulator.attributes[key].to_ndarray()
            for drop_id in range(particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])

    error = np.zeros(len(aerosol.modes))
    activated_fraction_S = np.zeros(len(aerosol.modes))
    activated_fraction_V = np.zeros(len(aerosol.modes))
    for j, mode in enumerate(aerosol.modes):
        activated_drops_j_S = 0
        activated_drops_j_V = 0
        RHmax = np.nanmax(np.asarray(output["RH"]))
        for i, volume in enumerate(output_attributes["volume"]):
            if j * n_sd_per_mode <= i < (j + 1) * n_sd_per_mode:
                if output_attributes["critical supersaturation"][i][-1] < RHmax:
                    activated_drops_j_S += output_attributes["multiplicity"][i][-1]
                if output_attributes["critical volume"][i][-1] < volume[-1]:
                    activated_drops_j_V += output_attributes["multiplicity"][i][-1]
        Nj = np.asarray(output_attributes["multiplicity"])[
            j * n_sd_per_mode : (j + 1) * n_sd_per_mode, -1
        ]
        max_multiplicity_j = np.max(Nj)
        sum_multiplicity_j = np.sum(Nj)
        error[j] = max_multiplicity_j / sum_multiplicity_j
        activated_fraction_S[j] = activated_drops_j_S / sum_multiplicity_j
        activated_fraction_V[j] = activated_drops_j_V / sum_multiplicity_j

    Output = namedtuple(
        "Output",
        [
            "profile",
            "attributes",
            "aerosol",
            "activated_fraction_S",
            "activated_fraction_V",
            "error",
        ],
    )
    return Output(
        profile=output,
        attributes=output_attributes,
        aerosol=aerosol,
        activated_fraction_S=activated_fraction_S,
        activated_fraction_V=activated_fraction_V,
        error=error,
    )
