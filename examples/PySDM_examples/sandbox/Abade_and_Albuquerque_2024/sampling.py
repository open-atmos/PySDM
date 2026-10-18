import numpy as np
from PySDM.initialisation.spectra import Lognormal


def sampling(
    *,
    n_sd,
    freezing_inp_frac,
    rng,
    singular,
    freezing_inp_dry_radius,
    attributes,
    formulae,
):
    insoluble_surface_area = formulae.trivia.sphere_surface(
        diameter=2 * freezing_inp_dry_radius
    )
    n_inp = int(n_sd * freezing_inp_frac)

    attributes["freezing temperature" if singular else "immersed surface area"] = (
        rng.permutation(
            np.pad(
                (
                    formulae.freezing_temperature_spectrum.invcdf(
                        cdf=rng.uniform(low=0, high=1, size=n_inp),
                        A_insol=insoluble_surface_area,
                    )
                    if singular
                    else np.full(n_inp, insoluble_surface_area)
                ),
                (0, n_sd - n_inp),
                mode="constant",
                constant_values=(
                    formulae.constants.HOMOGENEOUS_FREEZING_THRESHOLD if singular else 0
                ),
            )
        )
    )


def sampling_new(
    *,
    n_sd,
    freezing_inp_frac,
    rng,
    singular,
    freezing_inp_dry_radius,
    attributes,
    formulae,
    inp_surface_sigma=1.0,
):
    if not np.isfinite(inp_surface_sigma) or inp_surface_sigma < 1:
        raise ValueError("inp_surface_sigma must be finite and >= 1.")

    surface_median = formulae.trivia.sphere_surface(
        diameter=2 * freezing_inp_dry_radius
    )
    n_inp = int(n_sd * freezing_inp_frac)
    surface_spectrum = Lognormal(
        norm_factor=1,
        m_mode=surface_median,
        s_geom=inp_surface_sigma,
    )
    insoluble_surface_area = surface_spectrum.percentiles(
        rng.uniform(low=0, high=1, size=n_inp)
    )

    attributes["freezing temperature" if singular else "immersed surface area"] = (
        rng.permutation(
            np.pad(
                (
                    formulae.freezing_temperature_spectrum.invcdf(
                        cdf=rng.uniform(low=0, high=1, size=n_inp),
                        A_insol=insoluble_surface_area,
                    )
                    if singular
                    else insoluble_surface_area
                ),
                (0, n_sd - n_inp),
                mode="constant",
                constant_values=(
                    formulae.constants.HOMOGENEOUS_FREEZING_THRESHOLD if singular else 0
                ),
            )
        )
    )
