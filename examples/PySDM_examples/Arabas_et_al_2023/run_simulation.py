import numpy as np


def update_thermo(particulator, T):
    env = particulator.environment
    svp = particulator.formulae.saturation_vapour_pressure
    const = particulator.formulae.constants

    env["T"] = T
    env["a_w_ice"] = svp.ice_Celsius(T - const.T0) / svp.pvs_Celsius(T - const.T0)


def run_simulation(particulator, temperature_profile, n_steps):
    output = {
        "products": {k: [] for k in particulator.products.keys()},
        "attributes": [],
    }
    for step in range(n_steps + 1):
        if step != 0:
            update_thermo(
                particulator, T=temperature_profile((step - 0.5) * particulator.dt)
            )
            particulator.run(step - particulator.n_steps)
            update_thermo(particulator, T=temperature_profile(step * particulator.dt))
            output["frozen"].append(particulator.attributes["volume"].to_ndarray() < 0)
        else:
            output["spectrum"] = {}
            output["frozen"] = [np.full(particulator.n_sd, False)]
            for k in ("multiplicity", "freezing temperature", "immersed surface area"):
                if k in particulator.attributes:
                    output["spectrum"][k] = particulator.attributes[k].to_ndarray()
        for k, v in particulator.products.items():
            output["products"][k].append(v.get() + 0)
    return output
